import os
import re
import uuid
import shutil
import tempfile
import pydicom
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
from pydicom.tag import Tag
import datetime

# ==== CONFIGURATION ====
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'.dcm'}
PER_PAGE = 10

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')  # Use environment variable in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.globals.update(zip=zip)

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# In-memory report index (for demo)
REPORT_INDEX = []

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

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

def format_date(dcm_date):
    """Convert DICOM date string (YYYYMMDD) to ISO 8601 format (YYYY-MM-DD)"""
    if isinstance(dcm_date, str) and len(dcm_date) == 8:
        try:
            if dcm_date.isdigit():
                return f"{dcm_date[:4]}-{dcm_date[4:6]}-{dcm_date[6:8]}"
            else:
                return dcm_date # Return original if not all digits (e.g. "N/A" or malformed)
        except ValueError:
            return dcm_date # Return original if error during slicing/formatting
    return str(dcm_date) # Return as string if not typical DICOM date format

# ==== DICOM DOSE REPORT EXTRACTION ====
def get_clean_value(dataset, tag_or_keyword, default='N/A'):
    """Safely extracts, cleans, and returns a DICOM tag's value as a string."""
    value_to_process = None
    try:
        raw_element_or_value = dataset.get(tag_or_keyword, None)

        if raw_element_or_value is None:
            return default

        # Check if we got a DataElement object or the value directly
        if isinstance(raw_element_or_value, pydicom.dataelem.DataElement):
            if raw_element_or_value.value is None: # Handles missing tag or empty value
                return default
            if hasattr(raw_element_or_value, 'VM') and raw_element_or_value.VM == 0: # Explicitly empty tag
                return default
            value_to_process = raw_element_or_value.value
        else:
            # Assume raw_element_or_value is the actual value (e.g., for keywords like 'Modality')
            value_to_process = raw_element_or_value
        
        if value_to_process is None: # Double check after potentially accessing .value
             return default

        if isinstance(value_to_process, pydicom.multival.MultiValue):
            return ', '.join(map(str, value_to_process)).strip()
        elif isinstance(value_to_process, (bytes, bytearray)):
            try:
                return value_to_process.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                try:
                    return value_to_process.decode('latin-1', errors='replace').strip()
                except Exception: # Fallback for unhandled decoding issues
                    return "Binary Data (Undecodable)"
        return str(value_to_process).strip()

    except Exception as e:
        print(f"Error getting value for {tag_or_keyword}: {e}")
        return default

def extract_dx_dose_info(ds):
    info = {}
    info['Patient ID'] = get_clean_value(ds, (0x0010, 0x0020))
    info['Study Date'] = format_date(get_clean_value(ds, (0x0008, 0x0020)))

    dx_tags = {
        'Body Part Examined': (0x0018, 0x0015), 'View Position': (0x0018, 0x5101),
        'Exposure (mAs)': (0x0018, 0x1152), 'kVp': (0x0018, 0x0060),
        'SID (mm)': (0x0018, 0x1110), 'Filter Type': (0x0018, 0x1160),
        'Grid': (0x0018, 0x1166), 'Focal Spot (mm)': (0x0018, 0x1190)
    }
    for display_name, tag_tuple in dx_tags.items():
        value_str = get_clean_value(ds, tag_tuple, f'N/A ({tag_tuple[0]:04x},{tag_tuple[1]:04x})')
        # Attempt numeric conversion for known DS types after getting clean string
        if display_name in ['Exposure (mAs)', 'kVp', 'SID (mm)', 'Focal Spot (mm)']:
            try:
                # Only try to float if not already an N/A string
                if not value_str.lower().startswith('n/a'):
                    info[display_name] = f"{round(float(value_str), 2):.2f}"
                else:
                    info[display_name] = value_str # Keep the N/A string with tag info
            except ValueError:
                info[display_name] = value_str # If float conversion fails, use original string
        else:
            info[display_name] = value_str
            
    dap_val_str = get_clean_value(ds, (0x0018, 0x115E), 'N/A (0018,115E)')
    if not dap_val_str.lower().startswith('n/a'):
        try:
            dap_dgycm2 = float(dap_val_str)
            info['DAP (dGy·cm²)'] = f"{dap_dgycm2:.2f}"
            info['DAP (µGy·m²)'] = f"{(dap_dgycm2 * 10):.2f}"
        except ValueError:
            info['DAP (dGy·cm²)'] = 'Invalid (0018,115E)'
            info['DAP (µGy·m²)'] = 'Invalid (0018,115E)'
    else:
        info['DAP (dGy·cm²)'] = dap_val_str
        info['DAP (µGy·m²)'] = dap_val_str

    entrance_dose_str = get_clean_value(ds, (0x0040, 0x8302), 'N/A (0040,8302)')
    if not entrance_dose_str.lower().startswith('n/a'):
        try:
            info['Entrance Dose (mGy)'] = f"{float(entrance_dose_str):.3f}"
        except ValueError:
            info['Entrance Dose (mGy)'] = 'Invalid (0040,8302)'
    else:
        info['Entrance Dose (mGy)'] = entrance_dose_str
        
    info.update(validate_dx_parameters(info)) # validate_dx_parameters should use the already processed values
    return info

def validate_dx_parameters(info): # This function assumes values in info are already strings
    validations = {}
    sid_str = info.get('SID (mm)', 'N/A')
    if not sid_str.lower().startswith('n/a'):
        try:
            sid = float(sid_str)
            validations['SID Status'] = 'Valid' if 700 <= sid <= 1800 else f'Outlier ({sid}mm)'
        except ValueError:
            validations['SID Status'] = 'Invalid SID value'
    else:
        validations['SID Status'] = 'N/A'

    exp_str = info.get('Exposure (mAs)', 'N/A')
    if not exp_str.lower().startswith('n/a'):
        try:
            exp = float(exp_str)
            validations['Exposure Status'] = 'Normal' if 0.1 <= exp <= 500 else f'Extreme ({exp}mAs)'
        except ValueError:
            validations['Exposure Status'] = 'Invalid exposure value'
    else:
        validations['Exposure Status'] = 'N/A'
    return validations

def extract_ct_dose_info(ds):
    info = { # Using get_clean_value for consistency and robustness
        'Patient ID': get_clean_value(ds, 'PatientID'),
        'Patient Age': get_clean_value(ds, 'PatientAge'),
        'Study Date': format_date(get_clean_value(ds, 'StudyDate')),
        'Manufacturer': get_clean_value(ds, 'Manufacturer'),
        'Study Description': get_clean_value(ds, 'StudyDescription'),
        'Series Description': get_clean_value(ds, 'SeriesDescription')
    }
    comments = ds.get('CommentsOnRadiationDose', '')
    dlp_dict = {}
    total_dlp = None # Initialize as None
    if comments and isinstance(comments, str): # Ensure comments is a string
        for match in re.finditer(r'Event=(\d+)\s*DLP=([\d.]+)', comments):
            try:
                event = int(match.group(1))
                dlp = float(match.group(2))
                dlp_dict[event] = dlp
            except ValueError:
                print(f"Warning: Could not parse DLP event in comments: {match.group(0)}")
        total_match = re.search(r'TotalDLP=([\d.]+)', comments)
        if total_match:
            try:
                total_dlp = float(total_match.group(1))
            except ValueError:
                print(f"Warning: Could not parse TotalDLP in comments: {total_match.group(1)}")
                total_dlp = None # Ensure it's None if parsing fails

    results = []
    if hasattr(ds, 'ExposureDoseSequence') and ds.ExposureDoseSequence: # Check if sequence exists and is not empty
        for idx, item in enumerate(ds.ExposureDoseSequence, 1):
            acq_type = getattr(item, 'AcquisitionType', None)
            ctdi_vol = getattr(item, 'CTDIvol', None)
            phantom = ''
            if hasattr(item, 'CTDIPhantomTypeCodeSequence') and item.CTDIPhantomTypeCodeSequence:
                seq = item.CTDIPhantomTypeCodeSequence
                if seq[0] and hasattr(seq[0], 'CodeValue'): # Check if first item exists
                    code = seq[0].CodeValue
                    if code in ['113690', '113701']:
                        phantom = 'Head 16cm'
                    elif code in ['113691', '113702']:
                        phantom = 'Body 32cm'
                    else:
                        phantom = seq[0].CodeMeaning if hasattr(seq[0], 'CodeMeaning') else code
            
            scan_type_str = 'N/A'
            if acq_type:
                acq_type_upper = str(acq_type).upper() # Ensure acq_type is string
                if acq_type_upper == 'SEQUENCED': scan_type_str = 'Axial'
                elif acq_type_upper == 'CONSTANT_ANGLE': scan_type_str = 'Scout/Localizer'
                elif acq_type_upper == 'SPIRAL': scan_type_str = 'Helical'
                else: scan_type_str = str(acq_type)
            
            ctdi_display = '' # Default to empty string
            if ctdi_vol is not None:
                try: ctdi_display = round(float(ctdi_vol), 2)
                except (ValueError, TypeError): ctdi_display = str(ctdi_vol) # Show raw if not float

            dlp_for_event = dlp_dict.get(idx, None)
            dlp_display = '' # Default to empty string
            if dlp_for_event is not None:
                try: dlp_display = round(float(dlp_for_event), 2)
                except (ValueError, TypeError): dlp_display = str(dlp_for_event)

            results.append({
                'Series': idx, 'Type': scan_type_str,
                'CTDIvol': ctdi_display if scan_type_str != 'Scout/Localizer' else '',
                'DLP': dlp_display if scan_type_str != 'Scout/Localizer' else '',
                'Phantom': phantom
            })
    return info, results, total_dlp

# ==== ROUTES ====
@app.route('/', methods=['GET', 'POST'])
def select_modality():
    modalities = ['CT', 'DX', 'MG'] # MG still placeholder
    if request.method == 'POST':
        selected_modality = request.form.get('modality')
        if selected_modality in modalities:
            
            # --- START: Added logic for clearing data ---
            # Only clear if the modality is actually changing or if it's a new selection
            # For this in-memory app, clearing every time a modality is confirmed via POST is simplest.
            if session.get('modality') != selected_modality or not REPORT_INDEX: # Or simply always clear on POST
                # 1. Clear the in-memory report index
                REPORT_INDEX.clear()
                print("REPORT_INDEX cleared.")

                # 2. Clear uploaded DICOM files from the UPLOAD_FOLDER
                # Use app.config['UPLOAD_FOLDER'] as it's set for the app instance
                clear_folder(app.config['UPLOAD_FOLDER'], extensions=['.dcm'])
                # 3. Clear generated plots from the STATIC_FOLDER
                # Be specific with extensions to avoid deleting other static assets
                clear_folder(STATIC_FOLDER, extensions=['.png'])
                flash_message = f"Modality set to {selected_modality}."
            else:
                # Modality re-selected is the same as current, no need to clear if data exists.
                # Or, if you always want to clear on any POST to this route:
                # (Then the above 'if' condition is not needed, just the clearing logic)
                flash_message = f"Modality re-confirmed as {selected_modality}."

            session['modality'] = selected_modality
            flash(flash_message, "info")
            return redirect(url_for('index')) # Redirect to upload page for the (newly) selected modality
            # --- END: Added logic for clearing data ---
        else:
            flash("Invalid modality selected. Please try again.", "error")
            # Stay on the select_modality page

    # For GET request or if POST fails validation before setting session
    return render_template('select_modality.html', modalities=modalities)

@app.route('/upload', methods=['GET'])
def index():
    modality = session.get('modality', None)
    if not modality:
        flash("Please select a modality first.", "warning")
        return redirect(url_for('select_modality'))
    return render_template('index.html', modality=modality)

@app.route('/process', methods=['POST'])
def process_files():
    modality = session.get('modality', None)
    if not modality:
        flash("Modality not selected. Please select a modality first.", "error")
        return redirect(url_for('select_modality'))

    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        flash("No files were selected for upload.", "warning")
        return redirect(url_for('index'))

    errors = []
    processed_count = 0
    
    for file in uploaded_files:
        if file.filename == '':
            continue
        if not allowed_file(file.filename):
            errors.append(f"File '{file.filename}' has an invalid extension. Only .dcm files are allowed.")
            continue
            
        filename = secure_filename(file.filename)
        unique_suffix = uuid.uuid4().hex[:8]
        save_name = f"{unique_suffix}_{filename}"
        temp_path = os.path.join(UPLOAD_FOLDER, save_name)
        
        try:
            file.save(temp_path)
            ds = pydicom.dcmread(temp_path, stop_before_pixels=True)
            
            # This call to get_clean_value should now work correctly
            file_modality_tag = get_clean_value(ds, 'Modality', 'UNKNOWN').upper() 
            if file_modality_tag != modality:
                errors.append(f"File '{filename}' is a {file_modality_tag} file, but {modality} was selected. Skipped.")
                if os.path.exists(temp_path): os.remove(temp_path) 
                continue

            # Use the extraction functions for primary data for consistency
            if modality == 'CT':
                main_info, _series_data, _total_dlp = extract_ct_dose_info(ds)
            elif modality == 'DX':
                main_info = extract_dx_dose_info(ds)
            else: # Should not happen if modality selection is validated
                errors.append(f"Unsupported modality '{modality}' for processing file '{filename}'.")
                if os.path.exists(temp_path): os.remove(temp_path)
                continue

            report_id = f"{unique_suffix}_{os.path.splitext(filename)[0]}"
            
            report_data = {
                'id': report_id, 
                'filename': filename,
                'save_name': save_name,
                'modality': modality, 
                # Get core fields from the already processed main_info
                'Patient ID': main_info.get('Patient ID', 'N/A'),
                'Study Date': main_info.get('Study Date', 'N/A'), # Already formatted
                'Raw Study Date': get_clean_value(ds, (0x0008,0x0020)) # Get raw for storage if needed
            }
            
            # Add modality-specific fields from main_info that aren't the core ones above
            if modality == 'CT':
                report_data.update({
                    'study_description': main_info.get('Study Description', 'N/A'),
                    'series_description': main_info.get('Series Description', 'N/A'),
                    'patient_age': main_info.get('Patient Age', 'N/A')
                })
            elif modality == 'DX':
                report_data.update({
                    'body_part': main_info.get('Body Part Examined', 'N/A'),
                    'view_position': main_info.get('View Position', 'N/A'),
                    'exposure': main_info.get('Exposure (mAs)', 'N/A'),
                    'kvp': main_info.get('kVp', 'N/A'),
                    'dap_value_ugy_m2': main_info.get('DAP (µGy·m²)', 'N/A') 
                })
            
            REPORT_INDEX.append(report_data)
            processed_count += 1
            
        except pydicom.errors.InvalidDicomError:
            errors.append(f"File '{filename}' is not a valid DICOM format or is corrupted.")
            if os.path.exists(temp_path): os.remove(temp_path)
        except Exception as e:
            errors.append(f"Error processing '{filename}': {str(e)} (Type: {type(e).__name__})")
            if os.path.exists(temp_path): os.remove(temp_path)
    
    if errors: flash('; '.join(errors), 'error')
    if processed_count > 0: flash(f"Successfully processed {processed_count} {modality} file(s).", 'success')
    elif not errors and processed_count == 0 and any(f.filename != '' for f in uploaded_files):
        flash("No files were processed successfully. Check error messages if any, or ensure modality selection matches file content.", "warning")

    return redirect(url_for('list_reports'))

@app.route('/reports')
def list_reports():
    page = int(request.args.get('page', 1))
    search_term = request.args.get('search', '').strip().lower() 
    
    modality_filter = session.get('modality')
    if not modality_filter:
        flash("Please select a modality to view reports.", "warning")
        return redirect(url_for('select_modality'))

    relevant_reports = [r for r in REPORT_INDEX if r['modality'] == modality_filter]
    
    if search_term:
        final_filtered_reports = []
        for r in relevant_reports:
            # Check base fields
            if search_term in r['filename'].lower() or \
               search_term in r.get('Patient ID', '').lower() or \
               search_term in r.get('Study Date', '').lower() or \
               search_term in r.get('Raw Study Date', '').lower():
                final_filtered_reports.append(r)
                continue # Already added, skip further checks for this report
            # Check modality-specific fields
            if r['modality'] == 'CT' and search_term in r.get('study_description', '').lower():
                final_filtered_reports.append(r)
            elif r['modality'] == 'DX' and \
                 (search_term in r.get('body_part', '').lower() or \
                  search_term in r.get('view_position', '').lower()):
                final_filtered_reports.append(r)
        # Remove duplicates if any were added by multiple conditions (unlikely with current logic but good practice)
        # Using a temporary list of ids to ensure uniqueness if reports could be complex objects
        # For this simple list of dicts, the continue above largely handles it.
        # If stricter uniqueness is needed:
        # seen_ids = set()
        # unique_filtered_reports = []
        # for report in final_filtered_reports:
        #    if report['id'] not in seen_ids:
        #        unique_filtered_reports.append(report)
        #        seen_ids.add(report['id'])
        # final_filtered_reports = unique_filtered_reports

    else:
        final_filtered_reports = relevant_reports
        
    total = len(final_filtered_reports)
    pages = (total + PER_PAGE - 1) // PER_PAGE if PER_PAGE > 0 else 1
    if page > pages and pages > 0 : page = pages # Handle page number out of bounds
    if page < 1: page = 1

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    reports_on_page = final_filtered_reports[start:end]

    study_descriptions = []
    if modality_filter == 'CT':
        study_descriptions = sorted(list(set(r['study_description'] for r in relevant_reports if r.get('study_description') and r.get('study_description','N/A') != 'N/A')))

    body_parts_examined = []
    if modality_filter == 'DX':
        body_parts_examined = sorted(list(set(r['body_part'] for r in relevant_reports if r.get('body_part') and r.get('body_part','N/A') != 'N/A')))

    return render_template(
        'report_list.html',
        reports=reports_on_page,
        study_descriptions=study_descriptions,
        body_parts_examined=body_parts_examined,
        page=page,
        pages=pages,
        search=search_term,
        total=total,
        current_modality=modality_filter
    )

@app.route('/delete_file/<report_id>', methods=['POST'])
def delete_file(report_id):
    report_to_delete = None
    report_index_to_remove = -1

    for i, report_item in enumerate(REPORT_INDEX):
        if report_item['id'] == report_id:
            report_to_delete = report_item
            report_index_to_remove = i
            break
            
    if not report_to_delete:
        flash(f"Report with ID '{report_id}' not found in index.", 'error')
        return redirect(url_for('list_reports'))
    
    if report_index_to_remove != -1:
        del REPORT_INDEX[report_index_to_remove]
    else:
        flash(f"Internal error: Report '{report_id}' found but index not determined for deletion.", 'error')
        return redirect(url_for('list_reports'))

    file_path = os.path.join(UPLOAD_FOLDER, report_to_delete['save_name'])
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            flash(f"File '{report_to_delete['filename']}' deleted successfully from disk and index.", 'success')
        except Exception as e:
            flash(f"Error deleting file '{report_to_delete['filename']}' from disk: {e}. Removed from index only.", 'error')
            # Consider if you want to re-add to REPORT_INDEX if disk deletion fails
            # For simplicity, we are keeping it removed from index.
    else:
        flash(f"File '{report_to_delete['filename']}' (saved as '{report_to_delete['save_name']}') not found on disk, but removed from index.", 'warning')
        
    # Redirect to the same page user was on, if possible, or to first page
    page = request.form.get('page', 1, type=int)
    search = request.form.get('search', '')
    return redirect(url_for('list_reports', page=page, search=search))


@app.route('/report/<report_id>')
def view_report(report_id):
    report_meta = next((r for r in REPORT_INDEX if r['id'] == report_id), None)
    if not report_meta:
        flash(f"Report with ID '{report_id}' not found.", 'error')
        return redirect(url_for('list_reports'))
    
    errors_list = []
    template_context = {
        'modality': report_meta['modality'],
        'filename': report_meta['filename'],
        'report_id': report_id,
        'info': None, 'dx_info': None, 'data': None, 'total_dlp': None,
        'errors': errors_list
    }
    
    try:
        dicom_file_path = os.path.join(UPLOAD_FOLDER, report_meta['save_name'])
        if not os.path.exists(dicom_file_path):
            msg = f"DICOM file '{report_meta['filename']}' (saved as '{report_meta['save_name']}') not found on disk."
            flash(msg, "error")
            errors_list.append(msg)
            # Populate with metadata if file is missing for partial view
            if report_meta['modality'] == 'CT': template_context['info'] = report_meta
            elif report_meta['modality'] == 'DX': template_context['dx_info'] = report_meta
            return render_template('results.html', **template_context)

        ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
        
        if report_meta['modality'] == 'CT':
            ct_info_details, ct_series_data, total_dlp_val = extract_ct_dose_info(ds)
            template_context['info'] = ct_info_details
            template_context['data'] = ct_series_data
            template_context['total_dlp'] = total_dlp_val
        elif report_meta['modality'] == 'DX':
            dx_info_details = extract_dx_dose_info(ds)
            template_context['dx_info'] = dx_info_details
        else:
            flash(f"Unsupported modality '{report_meta['modality']}' for detailed view.", "error")
            return redirect(url_for('list_reports')) # Or render results with an error
        
    except pydicom.errors.InvalidDicomError:
        msg = f"File '{report_meta['filename']}' is not a valid DICOM format or is corrupted."
        flash(msg, "error")
        errors_list.append(msg)
    except Exception as e:
        msg = f"Error reading report details for '{report_meta['filename']}': {e}"
        flash(msg, "error")
        errors_list.append(msg)
        print(f"Detailed error for {report_meta['filename']}: {type(e).__name__} - {e}") # Log full error to console

    # If info/dx_info was not populated due to error AFTER file check, ensure they are from report_meta
    if template_context['modality'] == 'CT' and not template_context['info']:
        template_context['info'] = report_meta 
    elif template_context['modality'] == 'DX' and not template_context['dx_info']:
        template_context['dx_info'] = report_meta
        
    return render_template('results.html', **template_context)


@app.route('/compare_dlp') # CT DLP Comparison
def compare_dlp():
    if session.get('modality') != 'CT':
        flash("DLP Comparison is for CT modality only.", "warning")
        return redirect(url_for('list_reports'))

    records = []
    ct_reports = [r for r in REPORT_INDEX if r['modality'] == 'CT']

    if not ct_reports:
        flash("No CT reports uploaded yet to compare DLP.", "info")
        return render_template('compare_dlp.html', plots=[], tables=[], study_descriptions=[], selected_study=None, current_modality='CT')

    for report_meta in ct_reports:
        try:
            # Check if physical file exists before trying to read
            dicom_file_path = os.path.join(UPLOAD_FOLDER, report_meta['save_name'])
            if not os.path.exists(dicom_file_path):
                print(f"Warning: File {report_meta['save_name']} for DLP comparison not found on disk.")
                continue

            ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
            comments = ds.get('CommentsOnRadiationDose', '')
            total_dlp_val = None
            if isinstance(comments, str): # Ensure comments is string
                match = re.search(r'TotalDLP=([\d.]+)', comments)
                if match:
                    try: total_dlp_val = float(match.group(1))
                    except ValueError: pass # total_dlp_val remains None
            
            if total_dlp_val is not None and report_meta.get('study_description', 'N/A') != 'N/A':
                records.append({
                    'Study Description': report_meta.get('study_description', '').strip(),
                    'Patient ID': report_meta.get('Patient ID', 'N/A'),
                    'report_id': report_meta['id'],
                    'filename': report_meta['filename'],
                    'Total DLP': total_dlp_val,
                    'study_date': report_meta.get('Study Date', '')
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for DLP comparison: {e}")
            continue 
    
    if not records:
        flash("No CT reports with valid DLP data found for comparison.", "info")
        return render_template('compare_dlp.html', plots=[], tables=[], study_descriptions=[], selected_study=None, current_modality='CT')

    df = pd.DataFrame(records)
    plots = []
    tables = []
    
    # Get unique study descriptions from successfully processed records for the dropdown
    study_descriptions_for_page = sorted(list(df['Study Description'].unique()))
    selected_study_filter = request.args.get('study_desc_filter')

    df_to_plot = df.copy() # Start with all data
    if selected_study_filter and selected_study_filter in study_descriptions_for_page:
        df_to_plot = df[df['Study Description'] == selected_study_filter].copy()
    
    if df_to_plot.empty:
        if selected_study_filter:
             flash(f"No data to plot for Study Description: {selected_study_filter}", "info")
        else:
             flash("No data available to generate DLP comparison plots/tables.", "info")

    for study_desc_group, group_data in df_to_plot.groupby('Study Description'):
        if group_data.empty or group_data['Total DLP'].isnull().all(): continue

        plt.figure(figsize=(max(8, len(group_data['Patient ID']) * 0.5 + 2), 6.5)) 
        plt.bar(group_data['Patient ID'], group_data['Total DLP'], color='#95b8d1', width=0.6)
        plt.xlabel('Patient ID', fontsize=12)
        plt.ylabel('Total DLP (mGy·cm)', fontsize=12)
        plt.title(f'Total DLP for {study_desc_group}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout(pad=1.5)
        safe_study_desc = re.sub(r'[^a-zA-Z0-9_.-]', '_', study_desc_group) # Allow dots and hyphens
        plot_filename = f'dlp_plot_{safe_study_desc}_{uuid.uuid4().hex[:6]}.png'
        plot_path = os.path.join(STATIC_FOLDER, plot_filename)
        try:
            plt.savefig(plot_path)
            plots.append({'study': study_desc_group, 'plot_url': url_for('static', filename=plot_filename)})
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        finally:
            plt.close() # Ensure plot is closed
        
        table_data = group_data[['Patient ID', 'study_date', 'Total DLP', 'report_id', 'filename']].to_dict(orient='records')
        tables.append({'study': study_desc_group, 'table': table_data})
        
    return render_template(
        'compare_dlp.html',
        plots=plots,
        tables=tables,
        study_descriptions=study_descriptions_for_page, 
        selected_study=selected_study_filter,
        current_modality='CT'
    )

@app.route('/export_excel_filtered') # CT DLP Export
def export_excel_filtered():
    if session.get('modality') != 'CT':
        flash("Excel export is for CT modality only.", "error")
        return redirect(url_for('list_reports'))
        
    study_desc_filter = request.args.get('study_desc') 
    if not study_desc_filter:
        flash("Please select a Study Description to filter for export.", "warning")
        return redirect(url_for('compare_dlp'))

    records = []
    ct_reports_filtered = [
        r for r in REPORT_INDEX 
        if r['modality'] == 'CT' and r.get('study_description', '').strip() == study_desc_filter
    ]

    if not ct_reports_filtered:
        flash(f"No reports found for Study Description: {study_desc_filter} to export.", "info")
        return redirect(url_for('compare_dlp'))

    for report_meta in ct_reports_filtered:
        try:
            dicom_file_path = os.path.join(UPLOAD_FOLDER, report_meta['save_name'])
            if not os.path.exists(dicom_file_path):
                print(f"Warning: File {report_meta['save_name']} for Excel export not found.")
                continue
            ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
            comments = ds.get('CommentsOnRadiationDose', '')
            total_dlp_val = None
            if isinstance(comments, str):
                match = re.search(r'TotalDLP=([\d.]+)', comments)
                if match:
                    try: total_dlp_val = float(match.group(1))
                    except ValueError: pass
            
            if total_dlp_val is not None:
                records.append({
                    'Study Description': report_meta.get('study_description', ''),
                    'Patient ID': report_meta.get('Patient ID', 'N/A'),
                    'Filename': report_meta.get('filename', ''),
                    'Modality': report_meta.get('modality', ''),
                    'Total DLP (mGy.cm)': total_dlp_val,
                    'Study Date': report_meta.get('Study Date', '')
                })
            else: # Include row even if DLP is N/A for completeness of the selected study description
                 records.append({
                    'Study Description': report_meta.get('study_description', ''),
                    'Patient ID': report_meta.get('Patient ID', 'N/A'),
                    'Filename': report_meta.get('filename', ''),
                    'Modality': report_meta.get('modality', ''),
                    'Total DLP (mGy.cm)': 'N/A',
                    'Study Date': report_meta.get('Study Date', '')
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for Excel export: {e}")
            continue
            
    if not records: # Should be redundant if ct_reports_filtered check passed, but good failsafe
        flash(f"No data found to export for Study Description: {study_desc_filter}", "info")
        return redirect(url_for('compare_dlp'))

    df = pd.DataFrame(records)
    
    excel_output_ct = None # Initialize
    try:
        # Using BytesIO for better cloud/serverless compatibility for temp files
        from io import BytesIO
        output_stream = BytesIO()
        df.to_excel(output_stream, index=False, sheet_name='CT_DLP_Data')
        output_stream.seek(0) # Reset stream position
        
        safe_study_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', study_desc_filter)
        download_filename = f'CT_DLP_export_{safe_study_name}.xlsx'
        
        return send_file(
            output_stream,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        flash(f"Error generating Excel file: {e}", "error")
        return redirect(url_for('compare_dlp'))


# In app.py

@app.route('/compare_dap') # DX DAP Comparison
def compare_dap():
    if session.get('modality') != 'DX':
        flash("DAP Comparison is for DX modality only.", "warning")
        return redirect(url_for('list_reports'))

    records = []
    # Ensure dx_reports only contains reports for the current DX session
    # (This is already handled by how REPORT_INDEX is typically managed per session/modality selection)
    dx_reports = [r for r in REPORT_INDEX if r['modality'] == 'DX']

    if not dx_reports:
        flash("No DX reports uploaded yet to compare DAP.", "info")
        return render_template('compare_dap.html', plots=[], tables=[], body_parts_examined=[], selected_body_part=None, current_modality='DX')

    for report_meta in dx_reports:
        dap_str = report_meta.get('dap_value_ugy_m2', 'N/A')
        dap_value_num = None

        if dap_str != 'N/A' and not str(dap_str).lower().startswith('n/a'):
            try:
                dap_value_num = float(dap_str)
            except ValueError:
                print(f"Could not parse stored DAP value '{dap_str}' for {report_meta['filename']}")
                dap_value_num = None # Ensure it's None if parsing failed

        # Only add records that have a valid DAP and a valid Body Part
        if dap_value_num is not None and report_meta.get('body_part') and report_meta.get('body_part', 'N/A') != 'N/A':
            records.append({
                'Body Part Examined': report_meta.get('body_part', '').strip().upper(), # Ensure normalized
                'Patient ID': report_meta.get('Patient ID', 'N/A'),
                'report_id': report_meta['id'],
                'filename': report_meta['filename'],
                'DAP (µGy·m²)': dap_value_num,
                'study_date': report_meta.get('Study Date', '') # Assumed to be formatted
            })

    if not records:
        flash("No DX reports with valid DAP data and Body Part found for comparison.", "info")
        return render_template('compare_dap.html', plots=[], tables=[], body_parts_examined=[], selected_body_part=None, current_modality='DX')

    df_all_exams = pd.DataFrame(records) # DataFrame with all individual valid exams
    # Ensure DAP column is numeric for aggregation, coercing errors
    df_all_exams['DAP (µGy·m²)'] = pd.to_numeric(df_all_exams['DAP (µGy·m²)'], errors='coerce')
    df_all_exams.dropna(subset=['DAP (µGy·m²)'], inplace=True) # Remove rows where DAP is not numeric

    if df_all_exams.empty:
        flash("No valid numeric DAP data remaining after filtering for comparison.", "info")
        return render_template('compare_dap.html', plots=[], tables=[], body_parts_examined=[], selected_body_part=None, current_modality='DX')

    plots = []
    tables = []
    
    # Populate dropdown with unique body parts from all valid exams
    body_parts_for_page = sorted(list(df_all_exams['Body Part Examined'].unique()))
    selected_body_part_filter = request.args.get('body_part_filter')

    # Filter the DataFrame for plotting/tabling based on selection
    df_to_process = df_all_exams.copy()
    if selected_body_part_filter and selected_body_part_filter in body_parts_for_page:
        df_to_process = df_all_exams[df_all_exams['Body Part Examined'] == selected_body_part_filter].copy()
    
    if df_to_process.empty :
        if selected_body_part_filter:
            flash(f"No data found for Body Part: '{selected_body_part_filter}'.", "info")
        else: # No filter, but still empty after processing
            flash("No data available to generate DAP comparison plots/tables.", "info")
        # Still render the page with empty plots/tables but with dropdown populated
        return render_template(
            'compare_dap.html', plots=[], tables=[], 
            body_parts_examined=body_parts_for_page, 
            selected_body_part=selected_body_part_filter, 
            current_modality='DX'
        )

    # Group by Body Part for creating separate charts/tables
    for body_part_group, group_of_exams in df_to_process.groupby('Body Part Examined'):
        if group_of_exams.empty or group_of_exams['DAP (µGy·m²)'].isnull().all(): continue

        # ---- SOLUTION 2: Aggregate DAP per Patient ID for this Body Part Group ----
        # You can choose other aggregations like .max(), .sum(), .count()
        # .reset_index() converts the grouped Series back to a DataFrame
        patient_aggregated_dap = group_of_exams.groupby('Patient ID')['DAP (µGy·m²)'].mean().reset_index()
        aggregation_method_label = "Average" # Change if you use .max(), .sum(), etc.

        if patient_aggregated_dap.empty:
            continue # Skip if no data after aggregation for this group

        plt.figure(figsize=(max(8, len(patient_aggregated_dap['Patient ID']) * 0.5 + 2), 6.5))
        plt.bar(patient_aggregated_dap['Patient ID'], patient_aggregated_dap['DAP (µGy·m²)'], color='#2a9d8f', width=0.6)
        
        plt.xlabel('Patient ID', fontsize=12)
        plt.ylabel(f'{aggregation_method_label} DAP (µGy·m²)', fontsize=12) # Updated Y-axis label
        plt.title(f'{aggregation_method_label} DAP for {body_part_group}', fontsize=14, fontweight='bold') # Updated title
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout(pad=1.5)
        
        safe_body_part = re.sub(r'[^a-zA-Z0-9_.-]', '_', body_part_group)
        # Include aggregation method in plot filename for clarity if you experiment
        plot_filename = f'dap_plot_{safe_body_part}_{aggregation_method_label.lower()}_{uuid.uuid4().hex[:6]}.png'
        plot_path = os.path.join(STATIC_FOLDER, plot_filename)
        try:
            plt.savefig(plot_path)
            plots.append({
                'body_part': body_part_group, 
                'plot_url': url_for('static', filename=plot_filename),
                'aggregation_method': aggregation_method_label # Pass to template if needed
            })
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        finally:
            plt.close() # Crucial to close the figure
        
        # The table data will still show ALL individual exams for this body_part_group
        # The plot shows the aggregated summary, the table shows the detail.
        table_data = group_of_exams[['Patient ID', 'study_date', 'DAP (µGy·m²)', 'report_id', 'filename']].sort_values(by=['Patient ID', 'study_date']).to_dict(orient='records')
        tables.append({'body_part': body_part_group, 'table': table_data})
        
    return render_template(
        'compare_dap.html',
        plots=plots,
        tables=tables,
        body_parts_examined=body_parts_for_page, # For the dropdown
        selected_body_part=selected_body_part_filter,
        current_modality='DX'
    )
@app.route('/export_excel_dx_dap_filtered') # DX DAP Export
def export_excel_dx_dap_filtered():
    if session.get('modality') != 'DX':
        flash("Excel export is for DX modality only.", "error")
        return redirect(url_for('list_reports'))
        
    body_part_filter = request.args.get('body_part') 
    if not body_part_filter:
        flash("Please select a Body Part Examined to filter for export.", "warning")
        return redirect(url_for('compare_dap'))

    records = []
    dx_reports_filtered = [
        r for r in REPORT_INDEX 
        if r['modality'] == 'DX' and r.get('body_part', '').strip() == body_part_filter
    ]

    if not dx_reports_filtered:
        flash(f"No reports found for Body Part: {body_part_filter} to export.", "info")
        return redirect(url_for('compare_dap'))
        
    for report_meta in dx_reports_filtered:
        dap_str = report_meta.get('dap_value_ugy_m2', 'N/A')
        dap_value_num = None
        if dap_str != 'N/A' and not str(dap_str).lower().startswith('n/a'):
            try:
                dap_value_num = float(dap_str)
            except ValueError:
                dap_value_num = None # Keep as None if conversion fails
        
        # Include row if DAP is valid number, or even if N/A for completeness of selected body part
        records.append({
            'Body Part Examined': report_meta.get('body_part', ''),
            'Patient ID': report_meta.get('Patient ID', 'N/A'),
            'Filename': report_meta.get('filename', ''),
            'Modality': report_meta.get('modality', ''),
            'DAP (µGy·m²)': dap_value_num if dap_value_num is not None else 'N/A',
            'Study Date': report_meta.get('Study Date', '')
        })
            
    if not records: # Should be redundant
        flash(f"No data found to export for Body Part: {body_part_filter}", "info")
        return redirect(url_for('compare_dap'))

    df = pd.DataFrame(records)
    
    try:
        from io import BytesIO
        output_stream_dx = BytesIO()
        df.to_excel(output_stream_dx, index=False, sheet_name='DX_DAP_Data')
        output_stream_dx.seek(0)
        
        safe_body_part_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', body_part_filter)
        download_filename = f'DX_DAP_export_{safe_body_part_name}.xlsx'
        
        return send_file(
            output_stream_dx,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        flash(f"Error generating Excel file for DX DAP: {e}", "error")
        return redirect(url_for('compare_dap'))


@app.route('/search_patient', methods=['GET', 'POST'])
def search_patient():
    results = []
    patient_id_search = '' 
    current_modality = session.get('modality')

    if not current_modality:
        flash("Please select a modality first to search for patients.", "warning")
        return redirect(url_for('select_modality'))

    if request.method == 'POST':
        patient_id_search = request.form.get('patient_id', '').strip()
        if patient_id_search:
            reports_for_patient = [
                r for r in REPORT_INDEX 
                if r.get('Patient ID', '').strip() == patient_id_search and r.get('modality') == current_modality
            ]

            if not reports_for_patient:
                flash(f"No {current_modality} reports found for Patient ID: '{patient_id_search}'.", "info")

            for report_meta in reports_for_patient:
                entry = {
                    'report_id': report_meta['id'],
                    'study_date': report_meta.get('Study Date', 'N/A'), 
                    'filename': report_meta['filename']
                }

                if current_modality == 'CT':
                    entry['study_description'] = report_meta.get('study_description', 'N/A')
                    total_dlp_val = 'N/A' 
                    try:
                        dicom_file_path = os.path.join(UPLOAD_FOLDER, report_meta['save_name'])
                        if os.path.exists(dicom_file_path):
                            ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
                            comments = ds.get('CommentsOnRadiationDose', '')
                            if isinstance(comments, str):
                                match = re.search(r'TotalDLP=([\d.]+)', comments)
                                if match: total_dlp_val = float(match.group(1))
                        else: print(f"File not found for CT TotalDLP in patient search: {dicom_file_path}")
                    except Exception as e:
                        print(f"Error reading TotalDLP for '{report_meta['filename']}' in patient search: {e}")
                    entry['total_dlp'] = total_dlp_val
                
                elif current_modality == 'DX':
                    entry['body_part'] = report_meta.get('body_part', 'N/A')
                    entry['dap_value'] = report_meta.get('dap_value_ugy_m2', 'N/A')
                
                results.append(entry)
        elif request.method == 'POST' and not patient_id_search: # Submitted empty form
            flash("Please enter a Patient ID to search.", "warning")
            
    return render_template('search_patient.html', 
                           results=results, 
                           patient_id=patient_id_search, 
                           current_modality=current_modality)

@app.route('/compare_patient/<patient_id>')
def compare_patient(patient_id):
    studies_data = [] 
    current_modality = session.get('modality')

    if not current_modality:
        flash("Modality not set. Please select a modality first.", "error")
        return redirect(url_for('select_modality'))

    patient_id_stripped = patient_id.strip() # Ensure patient_id from URL is stripped
    reports_for_patient = [
        r for r in REPORT_INDEX 
        if r.get('Patient ID', '').strip() == patient_id_stripped and r.get('modality') == current_modality
    ]

    if not reports_for_patient:
        flash(f"No {current_modality} reports found for Patient ID: '{patient_id_stripped}' to compare.", "info")
        return render_template('compare_patient.html', patient_id=patient_id_stripped, plot_url=None, studies=[], current_modality=current_modality, value_label="Value")


    plot_generated = False
    plot_filename_patient = None 
    value_axis_label = "" 

    for report_meta in reports_for_patient:
        entry = {
            'report_id': report_meta['id'],
            'study_date': report_meta.get('Study Date', 'N/A'), 
            'filename': report_meta['filename']
        }
        numeric_value_for_plot = None

        if current_modality == 'CT':
            entry['study_description'] = report_meta.get('study_description', 'N/A')
            value_axis_label = 'Total DLP (mGy·cm)'
            entry['group_label'] = f"{entry['study_description'] or 'N/A'} ({entry['study_date']})"
            try:
                dicom_file_path = os.path.join(UPLOAD_FOLDER, report_meta['save_name'])
                if os.path.exists(dicom_file_path):
                    ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
                    comments = ds.get('CommentsOnRadiationDose', '')
                    if isinstance(comments, str):
                        match = re.search(r'TotalDLP=([\d.]+)', comments)
                        if match: numeric_value_for_plot = float(match.group(1))
                else: print(f"File not found for CT TotalDLP in patient comparison: {dicom_file_path}")
            except Exception as e:
                print(f"Error reading TotalDLP for plot '{report_meta['filename']}': {e}")
        
        elif current_modality == 'DX':
            entry['body_part'] = report_meta.get('body_part', 'N/A')
            value_axis_label = 'DAP (µGy·m²)'
            entry['group_label'] = f"{entry['body_part'] or 'N/A'} ({entry['study_date']})"
            dap_str = report_meta.get('dap_value_ugy_m2', 'N/A')
            if dap_str != 'N/A' and not str(dap_str).lower().startswith('n/a'):
                try: numeric_value_for_plot = float(dap_str)
                except ValueError: pass
        
        entry['value'] = numeric_value_for_plot
        if numeric_value_for_plot is not None:
             studies_data.append(entry)

    if not studies_data: 
        flash(f"No comparable numeric data found for Patient '{patient_id_stripped}' in {current_modality} modality.", "info")
        return render_template('compare_patient.html', patient_id=patient_id_stripped, plot_url=None, studies=reports_for_patient, current_modality=current_modality, value_label=value_axis_label or "Value") # Show all reports for patient even if no plot

    plot_data_points = [s for s in studies_data if s.get('value') is not None and s.get('group_label')]

    if plot_data_points:
        plt.figure(figsize=(max(8, len(plot_data_points) * 0.6 + 2), 6.5))
        x_labels = [s['group_label'] for s in plot_data_points]
        y_values = [s['value'] for s in plot_data_points]
        
        # Ensure value_axis_label is set if plot_data_points is not empty
        if not value_axis_label and plot_data_points: # Fallback if not set by modality branches
            value_axis_label = "Value" 

        plt.bar(x_labels, y_values, color='#5eaaa8', width=0.6)
        plt.xlabel('Study Details (Date)', fontsize=12)
        plt.ylabel(value_axis_label, fontsize=12)
        plt.title(f'{value_axis_label} for Patient {patient_id_stripped}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout(pad=1.5)
        
        safe_patient_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', patient_id_stripped)
        plot_filename_patient = f'patient_plot_{safe_patient_id}_{current_modality.lower()}_{uuid.uuid4().hex[:6]}.png'
        plot_path_patient = os.path.join(STATIC_FOLDER, plot_filename_patient)
        try:
            plt.savefig(plot_path_patient)
            plot_generated = True
        except Exception as e:
            print(f"Error saving patient comparison plot {plot_filename_patient}: {e}")
            flash(f"Error generating plot image: {e}", "error")
        finally:
            plt.close() # Always close the plot
    else: # This case should be less likely now due to earlier check on studies_data
        flash(f"No numeric data available to plot for Patient '{patient_id_stripped}' in {current_modality} modality.", "warning")

    return render_template('compare_patient.html', 
                           patient_id=patient_id_stripped, 
                           plot_url=url_for('static', filename=plot_filename_patient) if plot_generated else None, 
                           studies=reports_for_patient, # Show all reports for this patient for context
                           current_modality=current_modality,
                           value_label=value_axis_label or "Value")


# ==== CLOUD DEPLOYMENT SETTINGS ====
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port) # Debug True for development