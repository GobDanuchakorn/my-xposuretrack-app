from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import pydicom
from werkzeug.utils import secure_filename
import re
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import shutil

# ==== CONFIGURATION ====
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'.dcm'}
PER_PAGE = 10

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.globals.update(zip=zip)  # Enable zip in Jinja2 templates

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# In-memory report index (for demo)
REPORT_INDEX = []

# ==== UTILITY FUNCTIONS ====

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder_path, extensions=None):
    """Delete all files (optionally only with certain extensions) in a folder."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if extensions is None or any(filename.endswith(ext) for ext in extensions):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

# ==== DICOM DOSE REPORT EXTRACTION ====

def extract_ct_dose_info(ds):
    info = {
        'Patient ID': ds.get('PatientID', 'N/A'),
        'Patient Age': ds.get('PatientAge', 'N/A'),
        'Study Date': ds.get('StudyDate', 'N/A'),
        'Manufacturer': ds.get('Manufacturer', 'N/A'),
        'Study Description': ds.get('StudyDescription', 'N/A'),
        'Series Description': ds.get('SeriesDescription', 'N/A')
    }
    comments = ds.get('CommentsOnRadiationDose', '')
    dlp_dict = {}
    total_dlp = None
    if comments:
        for match in re.finditer(r'Event=(\d+)\s*DLP=([\d.]+)', comments):
            event = int(match.group(1))
            dlp = float(match.group(2))
            dlp_dict[event] = dlp
        total_match = re.search(r'TotalDLP=([\d.]+)', comments)
        if total_match:
            total_dlp = float(total_match.group(1))

    results = []
    if hasattr(ds, 'ExposureDoseSequence'):
        for idx, item in enumerate(ds.ExposureDoseSequence, 1):
            acq_type = getattr(item, 'AcquisitionType', None)
            ctdi_vol = getattr(item, 'CTDIvol', None)
            phantom = ''
            if hasattr(item, 'CTDIPhantomTypeCodeSequence'):
                seq = item.CTDIPhantomTypeCodeSequence
                if seq and hasattr(seq[0], 'CodeValue'):
                    code = seq[0].CodeValue
                    if code in ['113690', '113701']:
                        phantom = 'Head 16'
                    elif code in ['113691', '113702']:
                        phantom = 'Body 32'
                    else:
                        phantom = seq[0].CodeMeaning if hasattr(seq[0], 'CodeMeaning') else ''
            if acq_type:
                if acq_type.upper() == 'SEQUENCED':
                    scan_type = 'Axial'
                elif acq_type.upper() == 'CONSTANT_ANGLE':
                    scan_type = 'Scout'
                elif acq_type.upper() == 'SPIRAL':
                    scan_type = 'Helical'
                else:
                    scan_type = acq_type
            else:
                scan_type = 'N/A'
            ctdi_display = round(float(ctdi_vol), 2) if ctdi_vol is not None else ''
            dlp = dlp_dict.get(idx, '')
            dlp_display = round(float(dlp), 2) if dlp != '' else ''
            results.append({
                'Series': idx,
                'Type': scan_type,
                'CTDIvol': ctdi_display if scan_type != 'Scout' else '',
                'DLP': dlp_display if scan_type != 'Scout' else '',
                'Phantom': phantom
            })
    return info, results, total_dlp

# ==== ROUTES ====

@app.route('/', methods=['GET', 'POST'])
def select_modality():
    modalities = ['CT', 'DX', 'MG']
    if request.method == 'POST':
        selected_modality = request.form.get('modality')
        if selected_modality:
            session['modality'] = selected_modality
            return redirect(url_for('index'))
    return render_template('select_modality.html', modalities=modalities)

@app.route('/upload', methods=['GET'])
def index():
    modality = session.get('modality', None)
    if not modality:
        return redirect(url_for('select_modality'))
    return render_template('index.html', modality=modality)

@app.route('/process', methods=['POST'])
def process_files():
    modality = session.get('modality', None)
    if not modality:
        return redirect(url_for('select_modality'))
    
    uploaded_files = request.files.getlist('files')
    errors = []
    for file in uploaded_files:
        if file.filename == '':
            continue
        if not allowed_file(file.filename):
            errors.append(f"File {file.filename} is not a valid DICOM file.")
            continue
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        save_name = f"{unique_id}_{filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, save_name)
        try:
            file.save(temp_path)
            try:
                ds = pydicom.dcmread(temp_path, stop_before_pixels=True)
                study_description = ds.get('StudyDescription', 'N/A')
                patient_id = ds.get('PatientID', 'N/A')
            except Exception:
                study_description = 'N/A'
                patient_id = 'N/A'
            REPORT_INDEX.append({
                'id': unique_id,
                'filename': filename,
                'save_name': save_name,
                'modality': modality,
                'study_description': study_description,
                'Patient ID': patient_id
            })
        except Exception as e:
            errors.append(f"Error processing {filename}: {str(e)}")
    return redirect(url_for('list_reports'))

@app.route('/reports')
def list_reports():
    page = int(request.args.get('page', 1))
    search = request.args.get('search', '').strip().lower()
    filtered_reports = REPORT_INDEX
    if search:
        filtered_reports = [
            r for r in REPORT_INDEX
            if search in r['filename'].lower() or search in r['study_description'].lower()
        ]
    total = len(filtered_reports)
    pages = (total + PER_PAGE - 1) // PER_PAGE
    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    reports = filtered_reports[start:end]
    study_descriptions = []
    for report in REPORT_INDEX:
        if report['modality'] == 'CT' and report['study_description'] not in study_descriptions:
            study_descriptions.append(report['study_description'])
    return render_template(
        'report_list.html',
        reports=reports,
        study_descriptions=study_descriptions,
        page=page,
        pages=pages,
        search=search,
        total=total
    )

# ========== ลบไฟล์ที่อัปโหลด ==========
@app.route('/delete_file/<report_id>', methods=['POST'])
def delete_file(report_id):
    report = next((r for r in REPORT_INDEX if r['id'] == report_id), None)
    if not report:
        return "File not found", 404
    file_path = os.path.join(UPLOAD_FOLDER, report['save_name'])
    if os.path.exists(file_path):
        os.remove(file_path)
    REPORT_INDEX.remove(report)
    return redirect(url_for('list_reports'))

@app.route('/report/<report_id>')
def view_report(report_id):
    report = next((r for r in REPORT_INDEX if r['id'] == report_id), None)
    if not report:
        return "Report not found", 404
    try:
        ds = pydicom.dcmread(os.path.join(UPLOAD_FOLDER, report['save_name']), stop_before_pixels=True)
        info, data, total_dlp = extract_ct_dose_info(ds)
        return render_template('results.html', info=info, data=data, errors=[], total_dlp=total_dlp, modality=report['modality'], filename=report['filename'])
    except Exception as e:
        return f"Error reading report: {e}", 500

@app.route('/compare_dlp')
def compare_dlp():
    records = []
    for report in REPORT_INDEX:
        records.append({
            'Study Description': report.get('study_description', '').strip(),
            'Patient ID': report.get('Patient ID', 'N/A'),
            'report_id': report['id'],
            'filename': report['filename'],
            'save_name': report['save_name'],
            'modality': report['modality']
        })
        
    # Process DLP and study date
    for record in records:
        try:
            ds = pydicom.dcmread(os.path.join(UPLOAD_FOLDER, record['save_name']), stop_before_pixels=True)
            comments = ds.get('CommentsOnRadiationDose', '')
            total_dlp = None
            match = re.search(r'TotalDLP=([\d.]+)', comments)
            if match:
                total_dlp = float(match.group(1))
            record['Total DLP'] = total_dlp
            study_date = ds.get('StudyDate', '')
            if study_date and len(study_date) == 8:
                study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
            record['study_date'] = study_date
        except Exception:
            record['Total DLP'] = None
            record['study_date'] = ''
            
    # Filter valid records
    records = [r for r in records if r['Study Description'] and r['Total DLP'] is not None]
    
    if not records:
        return render_template('compare_dlp.html', plots=[], tables=[])

    # Prepare data for template
    df = pd.DataFrame(records)
    plots = []
    tables = []
    
    # Get unique study descriptions for dropdown
    study_descriptions = sorted({r['Study Description'] for r in records})
    selected_study = request.args.get('study_desc')

    # Group by study description
    for study, group in df.groupby('Study Description'):
        # Generate plot
        plt.figure(figsize=(8, 5))
        plt.bar(group['Patient ID'], group['Total DLP'], color='#95b8d1')
        plt.xlabel('Patient ID')
        plt.ylabel('Total DLP (mGy·cm)')
        plt.title(f'Total DLP for {study}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        safe_study = re.sub(r'[^a-zA-Z0-9]', '_', study)
        plot_filename = f'dlp_plot_{safe_study}.png'
        plot_path = os.path.join(STATIC_FOLDER, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        # Prepare data for template
        plots.append({'study': study, 'plot_url': f'static/{plot_filename}'})
        table = group[['Patient ID', 'study_date', 'Total DLP', 'report_id']].to_dict(orient='records')
        tables.append({'study': study, 'table': table})

    return render_template(
        'compare_dlp.html',
        plots=plots,
        tables=tables,
        study_descriptions=study_descriptions,
        selected_study=selected_study
    )


# ========== Export to Excel ==========
@app.route('/export_excel_filtered')
def export_excel_filtered():
    study_desc = request.args.get('study_desc')
    if not study_desc:
        return redirect(url_for('compare_dlp'))

    records = []
    filtered_reports = [r for r in REPORT_INDEX if r['study_description'] == study_desc]
    for report in filtered_reports:
        try:
            ds = pydicom.dcmread(os.path.join(UPLOAD_FOLDER, report['save_name']), stop_before_pixels=True)
            comments = ds.get('CommentsOnRadiationDose', '')
            total_dlp = None
            match = re.search(r'TotalDLP=([\d.]+)', comments)
            if match:
                total_dlp = float(match.group(1))
            study_date = ds.get('StudyDate', '')
            if study_date and len(study_date) == 8:
                study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
        except Exception:
            total_dlp = None
            study_date = ''
        records.append({
            'Study Description': report.get('study_description', ''),
            'Patient ID': report.get('Patient ID', 'N/A'),
            'Filename': report.get('filename', ''),
            'Modality': report.get('modality', ''),
            'Total DLP': total_dlp,
            'Study Date': study_date
        })
    if not records:
        return "No data to export", 400
    df = pd.DataFrame(records)
    safe_study = re.sub(r'[^a-zA-Z0-9]', '_', study_desc)
    file_path = os.path.join(STATIC_FOLDER, f'dlp_comparison_export_{safe_study}.xlsx')
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)


@app.route('/search_patient', methods=['GET', 'POST'])
def search_patient():
    results = []
    patient_id = ''
    if request.method == 'POST':
        patient_id = request.form.get('patient_id', '').strip()
        if patient_id:
            for report in REPORT_INDEX:
                if report.get('Patient ID', '').strip() == patient_id:
                    try:
                        ds = pydicom.dcmread(os.path.join(UPLOAD_FOLDER, report['save_name']), stop_before_pixels=True)
                        comments = ds.get('CommentsOnRadiationDose', '')
                        total_dlp = None
                        match = re.search(r'TotalDLP=([\d.]+)', comments)
                        if match:
                            total_dlp = float(match.group(1))
                        study_date = ds.get('StudyDate', '')
                        if study_date and len(study_date) == 8:
                            study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
                        results.append({
                            'study_description': report.get('study_description', ''),
                            'total_dlp': total_dlp,
                            'report_id': report['id'],
                            'study_date': study_date
                        })
                    except Exception:
                        continue
    return render_template('search_patient.html', results=results, patient_id=patient_id)

@app.route('/compare_patient/<patient_id>')
def compare_patient(patient_id):
    studies = []
    for report in REPORT_INDEX:
        if report.get('Patient ID', '').strip() == patient_id:
            try:
                ds = pydicom.dcmread(os.path.join(UPLOAD_FOLDER, report['save_name']), stop_before_pixels=True)
                comments = ds.get('CommentsOnRadiationDose', '')
                total_dlp = None
                match = re.search(r'TotalDLP=([\d.]+)', comments)
                if match:
                    total_dlp = float(match.group(1))
                # --- Extract and format Study Date ---
                study_date = ds.get('StudyDate', '')
                if study_date and len(study_date) == 8:
                    study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
                studies.append({
                    'study_description': report.get('study_description', ''),
                    'total_dlp': total_dlp,
                    'report_id': report['id'],
                    'study_date': study_date
                })
            except Exception:
                continue
    if not studies:
        return render_template('compare_patient.html', patient_id=patient_id, plot_url=None, studies=[])
    plt.figure(figsize=(8,5))
    x = [s['study_description'] for s in studies]
    y = [s['total_dlp'] for s in studies]
    plt.bar(x, y, color='#95b8d1')
    plt.xlabel('Study Description')
    plt.ylabel('Total DLP (mGy·cm)')
    plt.title(f'Total DLP for Patient {patient_id}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = f'patient_dlp_{patient_id}.png'
    plot_path = os.path.join(STATIC_FOLDER, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return render_template('compare_patient.html', patient_id=patient_id, plot_url=f'static/{plot_filename}', studies=studies)

import os
port = int(os.environ.get("PORT", 8080))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
