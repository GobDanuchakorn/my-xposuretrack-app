import os
import re
import uuid
import shutil
import tempfile
import pydicom
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
from pydicom.tag import Tag 
import datetime
from io import BytesIO 

# ==============================================================================
# ==== CONFIGURATION & APP INITIALIZATION ====
# ==============================================================================

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'.dcm'}
PER_PAGE = 20 

# Target DAP unit and keys for DX modality
TARGET_DAP_UNIT_LABEL_DX = 'DAP (mGy·cm²)'
TARGET_DAP_STORAGE_KEY_DX = 'dap_value_mgy_cm2' 
DAP_CONVERSION_FACTOR_FROM_DGY_CM2_TO_MGY_CM2 = 100

# Target Organ Dose unit and keys for MG modality
TARGET_ORGAN_DOSE_UNIT_LABEL_MG = 'Organ Dose (mGy)'
TARGET_ORGAN_DOSE_STORAGE_KEY_MG = 'organ_dose_mgy' 

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-please-change') 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100000 * 1024 * 1024  

app.jinja_env.globals.update(zip=zip)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True) 

REPORT_INDEX = []

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.datetime.now().year}

# ==============================================================================
# ==== UTILITY FUNCTIONS ====
# ==============================================================================

def allowed_file(filename):
    return '.' in filename and \
           os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder_path, extensions_to_delete=None):
    if not os.path.exists(folder_path):
        print(f"Warning: Folder to clear does not exist: {folder_path}")
        return
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if extensions_to_delete is None or \
               any(filename.lower().endswith(ext) for ext in extensions_to_delete):
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path) 
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    except Exception as e:
        print(f"Error listing directory {folder_path} for clearing: {e}")

def format_date(dcm_date_str):
    if isinstance(dcm_date_str, str) and len(dcm_date_str) == 8 and dcm_date_str.isdigit():
        try:
            return f"{dcm_date_str[:4]}-{dcm_date_str[4:6]}-{dcm_date_str[6:8]}"
        except ValueError: 
            return dcm_date_str 
    return str(dcm_date_str) 

def get_clean_value(dataset, tag_or_keyword, default='N/A'):
    value_to_process = None
    try:
        raw_element_or_value = dataset.get(tag_or_keyword, None)
        if raw_element_or_value is None:
            return default
        if isinstance(raw_element_or_value, pydicom.dataelem.DataElement):
            if raw_element_or_value.value is None:
                return default
            if hasattr(raw_element_or_value, 'VM') and raw_element_or_value.VM == 0:
                return default 
            value_to_process = raw_element_or_value.value
        else:
            value_to_process = raw_element_or_value 
        if value_to_process is None:
             return default
        if isinstance(value_to_process, pydicom.multival.MultiValue):
            return ', '.join(map(str, value_to_process)).strip()
        elif isinstance(value_to_process, (bytes, bytearray)):
            try: 
                return value_to_process.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                try: return value_to_process.decode('latin-1', errors='replace').strip()
                except: return "Binary Data (Undecodable)"
        return str(value_to_process).strip()
    except Exception as e:
        print(f"Error getting value for tag/keyword '{tag_or_keyword}': {type(e).__name__} - {e}")
        return default

# ==============================================================================
# ==== DICOM DOSE REPORT EXTRACTION FUNCTIONS ====
# ==============================================================================

def extract_dx_dose_info(ds):
    info = {}
    info['Patient ID'] = get_clean_value(ds, (0x0010, 0x0020))
    info['Study Date'] = format_date(get_clean_value(ds, (0x0008, 0x0020)))
    raw_body_part = get_clean_value(ds, (0x0018, 0x0015))
    if raw_body_part and raw_body_part.lower() != 'n/a' and not raw_body_part.lower().startswith('n/a ('):
        info['Body Part Examined'] = raw_body_part.strip().upper()
    else:
        info['Body Part Examined'] = 'N/A'
    dx_tags_to_extract = {
        'View Position': (0x0018, 0x5101),
        'Exposure (mAs)': (0x0018, 0x1152), 'kVp': (0x0018, 0x0060),
        'SID (mm)': (0x0018, 0x1110), 'Filter Type': (0x0018, 0x1160),
        'Grid': (0x0018, 0x1166), 'Focal Spot (mm)': (0x0018, 0x1190)
    }
    for display_name, tag_tuple in dx_tags_to_extract.items():
        value_str = get_clean_value(ds, tag_tuple, f'N/A ({tag_tuple[0]:04x},{tag_tuple[1]:04x})')
        if display_name in ['Exposure (mAs)', 'kVp', 'SID (mm)', 'Focal Spot (mm)']:
            try:
                if not value_str.lower().startswith('n/a'):
                    info[display_name] = f"{round(float(value_str), 2):.2f}"
                else:
                    info[display_name] = value_str
            except ValueError: 
                info[display_name] = f"Invalid: {value_str}"
        else:
            info[display_name] = value_str
    dap_val_str_dgycm2 = get_clean_value(ds, (0x0018, 0x115E), 'N/A')
    dap_tag_id_str = '(0018,115E)'
    info['DAP (dGy·cm²)'] = f'N/A {dap_tag_id_str}' 
    info[TARGET_DAP_UNIT_LABEL_DX] = f'N/A {dap_tag_id_str}' 
    if not dap_val_str_dgycm2.lower().startswith('n/a'):
        try:
            dap_dgycm2 = float(dap_val_str_dgycm2)
            info['DAP (dGy·cm²)'] = f"{dap_dgycm2:.2f}"
            info[TARGET_DAP_UNIT_LABEL_DX] = f"{(dap_dgycm2 * DAP_CONVERSION_FACTOR_FROM_DGY_CM2_TO_MGY_CM2):.2f}"
        except ValueError:
            info['DAP (dGy·cm²)'] = f"Invalid {dap_tag_id_str}: {dap_val_str_dgycm2}"
            info[TARGET_DAP_UNIT_LABEL_DX] = f"Invalid {dap_tag_id_str}: {dap_val_str_dgycm2}"
    entrance_dose_str = get_clean_value(ds, (0x0040, 0x8302), 'N/A')
    entrance_dose_tag_id_str = '(0040,8302)'
    info['Entrance Dose (mGy)'] = f'N/A {entrance_dose_tag_id_str}'
    if not entrance_dose_str.lower().startswith('n/a'):
        try:
            info['Entrance Dose (mGy)'] = f"{float(entrance_dose_str):.3f}"
        except ValueError:
            info['Entrance Dose (mGy)'] = f"Invalid {entrance_dose_tag_id_str}: {entrance_dose_str}"
    info.update(validate_dx_parameters(info))
    return info

def validate_dx_parameters(info):
    validations = {}
    sid_str = info.get('SID (mm)', 'N/A')
    if not sid_str.lower().startswith('n/a') and not sid_str.lower().startswith('invalid'):
        try:
            sid = float(sid_str)
            validations['SID Status'] = 'Valid' if 700 <= sid <= 1800 else f'Outlier ({sid}mm)'
        except ValueError: 
            validations['SID Status'] = 'Invalid SID value'
    else:
        validations['SID Status'] = sid_str 
    exp_str = info.get('Exposure (mAs)', 'N/A')
    if not exp_str.lower().startswith('n/a') and not exp_str.lower().startswith('invalid'):
        try:
            exp = float(exp_str)
            validations['Exposure Status'] = 'Normal' if 0.1 <= exp <= 500 else f'Extreme ({exp}mAs)'
        except ValueError:
            validations['Exposure Status'] = 'Invalid exposure value'
    else:
        validations['Exposure Status'] = exp_str
    return validations

def extract_ct_dose_info(ds):
    info = {
        'Patient ID': get_clean_value(ds, 'PatientID'),
        'Patient Age': get_clean_value(ds, 'PatientAge'),
        'Study Date': format_date(get_clean_value(ds, 'StudyDate')),
        'Manufacturer': get_clean_value(ds, 'Manufacturer'),
        'Study Description': get_clean_value(ds, 'StudyDescription'),
        'Series Description': get_clean_value(ds, 'SeriesDescription')
    }
    comments = get_clean_value(ds, 'CommentsOnRadiationDose', '')
    dlp_dict = {}
    total_dlp = None 
    if comments:
        for match in re.finditer(r'Event=(\d+)\s*DLP=([\d.]+)', comments):
            try:
                event = int(match.group(1))
                dlp = float(match.group(2))
                dlp_dict[event] = dlp
            except ValueError:
                print(f"Warning: Could not parse DLP event in CT comments: '{match.group(0)}'")
        total_match = re.search(r'TotalDLP=([\d.]+)', comments)
        if total_match:
            try:
                total_dlp = float(total_match.group(1))
            except ValueError:
                print(f"Warning: Could not parse TotalDLP in CT comments: '{total_match.group(1)}'")
                total_dlp = None 
    results = []
    if hasattr(ds, 'ExposureDoseSequence') and ds.ExposureDoseSequence:
        for idx, item in enumerate(ds.ExposureDoseSequence, 1):
            acq_type_val = getattr(item, 'AcquisitionType', None)
            ctdi_vol_val = getattr(item, 'CTDIvol', None)
            phantom = get_clean_value(item.CTDIPhantomTypeCodeSequence[0], 'CodeMeaning', '') \
                if hasattr(item, 'CTDIPhantomTypeCodeSequence') and item.CTDIPhantomTypeCodeSequence \
                else 'N/A'
            if hasattr(item, 'CTDIPhantomTypeCodeSequence') and item.CTDIPhantomTypeCodeSequence:
                code_val = get_clean_value(item.CTDIPhantomTypeCodeSequence[0], 'CodeValue', '')
                if code_val in ['113690', '113701']: phantom = 'Head 16cm'
                elif code_val in ['113691', '113702']: phantom = 'Body 32cm'
            scan_type_str = 'N/A'
            if acq_type_val:
                acq_type_upper = str(acq_type_val).upper()
                if acq_type_upper == 'SEQUENCED': scan_type_str = 'Axial'
                elif acq_type_upper == 'CONSTANT_ANGLE': scan_type_str = 'Scout/Localizer'
                elif acq_type_upper == 'SPIRAL': scan_type_str = 'Helical'
                else: scan_type_str = str(acq_type_val)
            ctdi_display = ''
            if ctdi_vol_val is not None:
                try: ctdi_display = round(float(ctdi_vol_val), 2)
                except (ValueError, TypeError): ctdi_display = str(ctdi_vol_val)
            dlp_for_event = dlp_dict.get(idx, None)
            dlp_display = ''
            if dlp_for_event is not None:
                try: dlp_display = round(float(dlp_for_event), 2)
                except (ValueError, TypeError): dlp_display = str(dlp_for_event)
            kvp_val = getattr(item, 'KVP', None)
            kvp_display = ''
            if kvp_val is not None:
                try: kvp_display = int(round(float(kvp_val)))
                except (ValueError, TypeError): kvp_display = str(kvp_val)
            tube_current_in_ua_val = getattr(item, 'XRayTubeCurrentInuA', None) 
            tube_current_display_ma = '' 
            if tube_current_in_ua_val is not None:
                try:
                    current_ma = float(tube_current_in_ua_val) / 1000.0
                    tube_current_display_ma = int(round(current_ma)) 
                except (ValueError, TypeError):
                    tube_current_display_ma = str(tube_current_in_ua_val) 
            exposure_time_val = getattr(item, 'ExposureTime', None)
            exposure_time_display = ''
            if exposure_time_val is not None:
                try: exposure_time_display = int(round(float(exposure_time_val)))
                except (ValueError, TypeError): exposure_time_display = str(exposure_time_val)
            pitch_factor_val = getattr(item, 'SpiralPitchFactor', None)
            pitch_factor_display = ''
            if pitch_factor_val is not None and scan_type_str == 'Helical':
                try: pitch_factor_display = round(float(pitch_factor_val), 3)
                except (ValueError, TypeError): pitch_factor_display = str(pitch_factor_val)
            elif scan_type_str != 'Helical':
                pitch_factor_display = 'N/A'
            results.append({
                'Series': idx, 'Type': scan_type_str,
                'CTDIvol': ctdi_display if scan_type_str != 'Scout/Localizer' else '',
                'DLP': dlp_display if scan_type_str != 'Scout/Localizer' else '',
                'Phantom': phantom,
                'KVP': kvp_display,
                'XRayTubeCurrent': tube_current_display_ma,
                'ExposureTime': exposure_time_display,
                'SpiralPitchFactor': pitch_factor_display
            })
    return info, results, total_dlp

def extract_mg_dose_info(ds):
    """Extracts and processes dose-related information for MG modality."""
    info = {}
    # Common patient and study information
    info['Patient ID'] = get_clean_value(ds, (0x0010, 0x0020))
    info['Patient Age'] = get_clean_value(ds, (0x0010, 0x1010)) 
    info['Study Date'] = format_date(get_clean_value(ds, (0x0008, 0x0020)))
    info['Study Description'] = get_clean_value(ds, (0x0008, 0x1030)) 
    info['Series Description'] = get_clean_value(ds, (0x0008, 0x103E)) 
    
    raw_body_part = get_clean_value(ds, (0x0018, 0x0015))
    if raw_body_part and raw_body_part.lower() != 'n/a' and not raw_body_part.lower().startswith('n/a ('):
        info['Body Part Examined'] = raw_body_part.strip().upper()
    else:
        info['Body Part Examined'] = 'N/A'
        
    info['View Position'] = get_clean_value(ds, (0x0018, 0x5101)) 

    # MG Specific technical parameters
    info['KVP'] = get_clean_value(ds, (0x0018, 0x0060), 'N/A') 
    info['Exposure (mAs-equivalent)'] = get_clean_value(ds, (0x0018, 0x1152), 'N/A') 
    info['Exposure Time (ms)'] = get_clean_value(ds, (0x0018, 0x1150), 'N/A') 
    info['X-Ray Tube Current (mA)'] = get_clean_value(ds, (0x0018, 0x1151), 'N/A') 

    # Entrance Dose
    entrance_dose_mgy_str = get_clean_value(ds, (0x0040, 0x8302), 'N/A') 
    entrance_dose_tag_id_str = '(0040,8302)'
    info['Entrance Dose (mGy)'] = f'N/A {entrance_dose_tag_id_str}'
    if not entrance_dose_mgy_str.lower().startswith('n/a'):
        try:
            info['Entrance Dose (mGy)'] = f"{float(entrance_dose_mgy_str):.3f}"
        except ValueError:
            info['Entrance Dose (mGy)'] = f"Invalid {entrance_dose_tag_id_str}: {entrance_dose_mgy_str}"

    # Organ Dose (Primary interest for MG)
    organ_dose_str = get_clean_value(ds, (0x0040, 0x0316), 'N/A') 
    organ_dose_tag_id_str = '(0040,0316)'
    
    info[TARGET_ORGAN_DOSE_UNIT_LABEL_MG] = f'N/A {organ_dose_tag_id_str}' 
    info[TARGET_ORGAN_DOSE_STORAGE_KEY_MG] = 'N/A' 

    if not organ_dose_str.lower().startswith('n/a'):
        try:
            organ_dose_val_raw = float(organ_dose_str)
            organ_dose_val_scaled = organ_dose_val_raw * 100
            info[TARGET_ORGAN_DOSE_UNIT_LABEL_MG] = f"{organ_dose_val_scaled:.4f}" 
            info[TARGET_ORGAN_DOSE_STORAGE_KEY_MG] = f"{organ_dose_val_scaled:.4f}"
        except ValueError:
            error_msg = f"Invalid {organ_dose_tag_id_str}: {organ_dose_str}"
            info[TARGET_ORGAN_DOSE_UNIT_LABEL_MG] = error_msg
            info[TARGET_ORGAN_DOSE_STORAGE_KEY_MG] = f"Invalid: {organ_dose_str}"
    info['Organ Exposed'] = get_clean_value(ds, (0x0040, 0x0318), 'N/A') 
    info['Body Part Thickness (mm)'] = get_clean_value(ds, (0x0018, 0x11A0), 'N/A') 
    info['Compression Force (N)'] = get_clean_value(ds, (0x0018, 0x11A2), 'N/A') 
    info['Filter Type'] = get_clean_value(ds, (0x0018, 0x1160), 'N/A') 
    info['Anode Target Material'] = get_clean_value(ds, (0x0018, 0x1191), 'N/A') 
    return info

# ==============================================================================
# ==== FLASK ROUTES ====
# ==============================================================================

@app.route('/', methods=['GET', 'POST'])
def select_modality():
    modalities = ['CT', 'DX', 'MG'] 
    if request.method == 'POST':
        selected_modality = request.form.get('modality')
        if selected_modality in modalities:
            current_session_modality = session.get('modality')
            if current_session_modality != selected_modality:
                REPORT_INDEX.clear()
                clear_folder(app.config['UPLOAD_FOLDER'], extensions_to_delete=['.dcm'])
                clear_folder(STATIC_FOLDER, extensions_to_delete=['.png']) 
                flash(f"Data cleared. Modality switched to {selected_modality}.", "info")
            else:
                 flash(f"Modality {selected_modality} re-confirmed.", "info")
            session['modality'] = selected_modality
            return redirect(url_for('index'))
        else:
            flash("Invalid modality selected. Please try again.", "error")
    return render_template('select_modality.html', modalities=modalities)

@app.route('/upload', methods=['GET'])
def index():
    modality = session.get('modality')
    if not modality:
        flash("Please select a modality first.", "warning")
        return redirect(url_for('select_modality'))
    return render_template('index.html', modality=modality)

@app.route('/process', methods=['POST'])
def process_files():
    modality = session.get('modality')
    if not modality:
        flash("Modality not selected. Please select a modality first.", "error")
        return redirect(url_for('select_modality'))
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        flash("No files were selected for upload.", "warning")
        return redirect(url_for('index'))
    errors, processed_count = [], 0
    for file_storage in uploaded_files: 
        if not file_storage.filename: continue
        if not allowed_file(file_storage.filename):
            errors.append(f"File '{file_storage.filename}' has an invalid extension. Only .dcm allowed.")
            continue
        original_filename = secure_filename(file_storage.filename)
        unique_suffix = uuid.uuid4().hex[:8]
        save_name = f"{unique_suffix}_{original_filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        try:
            file_storage.save(temp_path)
            ds = pydicom.dcmread(temp_path, stop_before_pixels=True)
            file_actual_modality = get_clean_value(ds, 'Modality', 'UNKNOWN').upper()
            if file_actual_modality != modality:
                expected_modality_in_file = modality
                if modality == 'MG' and file_actual_modality != 'MG': 
                     errors.append(f"File '{original_filename}' is {file_actual_modality}, but {modality} (MG) selected. Skipped.")
                else:
                    errors.append(f"File '{original_filename}' is {file_actual_modality}, but {modality} selected. Skipped.")
                if os.path.exists(temp_path): os.remove(temp_path)
                continue
            main_info = None 
            if modality == 'CT': main_info, _, _ = extract_ct_dose_info(ds)
            elif modality == 'DX': main_info = extract_dx_dose_info(ds)
            elif modality == 'MG': main_info = extract_mg_dose_info(ds)
            else:
                errors.append(f"Processing logic for modality '{modality}' not implemented for '{original_filename}'.")
                if os.path.exists(temp_path): os.remove(temp_path)
                continue
            report_id = f"{unique_suffix}_{os.path.splitext(original_filename)[0]}"
            report_data = {
                'id': report_id, 'filename': original_filename, 'save_name': save_name, 'modality': modality,
                'Patient ID': main_info.get('Patient ID', 'N/A'),
                'Study Date': main_info.get('Study Date', 'N/A'), 
                'Raw Study Date': get_clean_value(ds, (0x0008,0x0020)) 
            }
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
                    TARGET_DAP_STORAGE_KEY_DX: main_info.get(TARGET_DAP_UNIT_LABEL_DX, 'N/A')
                })
            elif modality == 'MG':
                report_data.update({
                    'study_description': main_info.get('Study Description', 'N/A'),
                    'series_description': main_info.get('Series Description', 'N/A'),
                    'body_part': main_info.get('Body Part Examined', 'N/A'), 
                    'view_position': main_info.get('View Position', 'N/A'),
                    TARGET_ORGAN_DOSE_STORAGE_KEY_MG: main_info.get(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'N/A'),
                    'organ_exposed': main_info.get('Organ Exposed', 'N/A')
                })
            REPORT_INDEX.append(report_data)
            processed_count += 1
        except pydicom.errors.InvalidDicomError:
            errors.append(f"File '{original_filename}' is invalid/corrupted.")
            if os.path.exists(temp_path): os.remove(temp_path)
        except Exception as e:
            errors.append(f"Error processing '{original_filename}': {type(e).__name__} - {str(e)}")
            if os.path.exists(temp_path): os.remove(temp_path)
    if errors: flash('; '.join(errors), 'error')
    if processed_count > 0: flash(f"Successfully processed {processed_count} {modality} file(s).", 'success')
    elif not errors and any(fs.filename for fs in uploaded_files): 
        flash("No files were processed successfully. Check errors or modality match.", "warning")
    return redirect(url_for('list_reports'))

@app.route('/reports')
def list_reports():
    modality_filter = session.get('modality')
    if not modality_filter:
        flash("Please select a modality to view reports.", "warning")
        return redirect(url_for('select_modality'))
    page = request.args.get('page', 1, type=int)
    search_term = request.args.get('search', '').strip().lower()
    sort_by = request.args.get('sort_by', 'Study Date') 
    sort_order = request.args.get('sort_order', 'desc') 
    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
    relevant_reports = [r for r in REPORT_INDEX if r['modality'] == modality_filter]
    if search_term:
        matching_report_ids = set()
        temp_filtered_reports = []
        for r in relevant_reports:
            matches_base = search_term in r['filename'].lower() or \
                           search_term in r.get('Patient ID', '').lower() or \
                           search_term in r.get('Study Date', '').lower() or \
                           search_term in r.get('Raw Study Date', '').lower()
            matches_modality_specific = False
            if r['modality'] == 'CT' and search_term in r.get('study_description', '').lower():
                matches_modality_specific = True
            elif r['modality'] == 'DX' and \
                 (search_term in r.get('body_part', '').lower() or \
                  search_term in r.get('view_position', '').lower()):
                matches_modality_specific = True
            elif r['modality'] == 'MG' and \
                 (search_term in r.get('study_description', '').lower() or \
                  search_term in r.get('body_part', '').lower() or \
                  search_term in r.get('view_position', '').lower() or \
                  search_term in r.get('organ_exposed', '').lower()):
                matches_modality_specific = True
            if (matches_base or matches_modality_specific) and r['id'] not in matching_report_ids:
                temp_filtered_reports.append(r)
                matching_report_ids.add(r['id'])
        final_filtered_reports = temp_filtered_reports
    else:
        final_filtered_reports = relevant_reports
    if final_filtered_reports: 
        is_reverse = (sort_order == 'desc')
        def sort_key_func(report):
            val = report.get(sort_by)
            if val is None: 
                return float('-inf') if sort_order == 'asc' else float('inf')
            if sort_by == TARGET_DAP_STORAGE_KEY_DX: 
                try:
                    val_str = str(val).lower()
                    if "invalid" in val_str or "n/a" in val_str:
                        return float('-inf') if sort_order == 'asc' else float('inf')
                    return float(val)
                except (ValueError, TypeError): 
                    return float('-inf') if sort_order == 'asc' else float('inf')
            if sort_by == TARGET_ORGAN_DOSE_STORAGE_KEY_MG:
                try:
                    val_str = str(val).lower()
                    if "invalid" in val_str or "n/a" in val_str:
                        return float('-inf') if sort_order == 'asc' else float('inf')
                    return float(val)
                except (ValueError, TypeError):
                    return float('-inf') if sort_order == 'asc' else float('inf')
            if sort_by == 'Study Date' and 'Raw Study Date' in report:
                 raw_date = report.get('Raw Study Date')
                 if raw_date and str(raw_date).isdigit() and len(str(raw_date)) == 8:
                     return raw_date
                 return val if val is not None else (float('-inf') if sort_order == 'asc' else float('inf'))
            if isinstance(val, str):
                return val.lower() 
            return val
        try:
            final_filtered_reports.sort(key=sort_key_func, reverse=is_reverse)
        except TypeError as e:
            flash(f"Could not sort by '{sort_by}'. Ensure data types are consistent or add specific handling. Error: {e}", "warning")
            pass
    total = len(final_filtered_reports)
    pages = (total + PER_PAGE - 1) // PER_PAGE if PER_PAGE > 0 else 1
    if page < 1: page = 1
    if page > pages and pages > 0: page = pages
    start_index = (page - 1) * PER_PAGE
    end_index = start_index + PER_PAGE
    reports_on_page = final_filtered_reports[start_index:end_index]
    study_descriptions_ct = []
    if modality_filter == 'CT':
        study_descriptions_ct = sorted(list(set(
            r['study_description'] for r in relevant_reports 
            if r.get('study_description') and r.get('study_description','N/A') != 'N/A'
        )))
    body_parts_dx_or_mg = [] 
    if modality_filter == 'DX' or modality_filter == 'MG': 
        body_parts_dx_or_mg = sorted(list(set(
            r['body_part'] for r in relevant_reports 
            if r.get('body_part') and r.get('body_part','N/A') != 'N/A'
        )))

    organ_exposed_mg = []
    mg_view_positions = [] # Initialize this list

    if modality_filter == 'MG':
        organ_exposed_mg = sorted(list(set(
            r['organ_exposed'] for r in relevant_reports
            if r.get('organ_exposed') and r.get('organ_exposed', 'N/A') != 'N/A'
        )))
        # Gather unique view positions for MG to be used in quick links
        mg_view_positions = sorted(list(set(
            r['view_position'] for r in relevant_reports
            if r.get('view_position') and r.get('view_position', 'N/A') != 'N/A'
        )))

    return render_template(
        'report_list.html', reports=reports_on_page, page=page, pages=pages,
        search=search_term, total=total, current_modality=modality_filter,
        study_descriptions=study_descriptions_ct, 
        body_parts_examined=body_parts_dx_or_mg,  
        organ_exposed_list=organ_exposed_mg, 
        mg_view_positions=mg_view_positions, # Pass the new list here
        TARGET_DAP_STORAGE_KEY_DX=TARGET_DAP_STORAGE_KEY_DX, 
        TARGET_ORGAN_DOSE_STORAGE_KEY_MG=TARGET_ORGAN_DOSE_STORAGE_KEY_MG,
        TARGET_ORGAN_DOSE_UNIT_LABEL_MG=TARGET_ORGAN_DOSE_UNIT_LABEL_MG,
        PER_PAGE=PER_PAGE,
        current_sort_by=sort_by,
        current_sort_order=sort_order
    )


@app.route('/delete_file/<report_id>', methods=['POST'])
def delete_file(report_id):
    report_to_delete, index_to_remove = None, -1
    for i, report_item in enumerate(REPORT_INDEX):
        if report_item['id'] == report_id:
            report_to_delete, index_to_remove = report_item, i
            break
    if not report_to_delete:
        flash(f"Report ID '{report_id}' not found.", 'error')
    else:
        del REPORT_INDEX[index_to_remove]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], report_to_delete['save_name'])
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                flash(f"File '{report_to_delete['filename']}' deleted successfully.", 'success')
            except Exception as e:
                flash(f"Error deleting file '{report_to_delete['filename']}' from disk: {e}. Removed from index.", 'error')
        else:
            flash(f"File '{report_to_delete['filename']}' (saved as '{report_to_delete['save_name']}') not on disk. Removed from index.", 'warning')
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
        'info': None, 
        'dx_info': None,
        'mg_info': None,
        'data': None, 
        'total_dlp': None,
        'errors': errors_list,
        'TARGET_ORGAN_DOSE_UNIT_LABEL_MG': TARGET_ORGAN_DOSE_UNIT_LABEL_MG
    }
    try:
        dicom_file_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
        if not os.path.exists(dicom_file_path):
            msg = f"DICOM file '{report_meta['filename']}' (saved as '{report_meta['save_name']}') not found on disk."
            errors_list.append(msg)
            if report_meta['modality'] == 'CT': template_context['info'] = report_meta
            elif report_meta['modality'] == 'DX': template_context['dx_info'] = report_meta
            elif report_meta['modality'] == 'MG': template_context['mg_info'] = report_meta
        else:
            ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
            if report_meta['modality'] == 'CT':
                ct_info_details, ct_series_data, total_dlp_val = extract_ct_dose_info(ds)
                template_context['info'] = ct_info_details
                template_context['data'] = ct_series_data
                template_context['total_dlp'] = total_dlp_val
            elif report_meta['modality'] == 'DX':
                dx_info_details = extract_dx_dose_info(ds)
                template_context['dx_info'] = dx_info_details
            elif report_meta['modality'] == 'MG':
                mg_info_details = extract_mg_dose_info(ds)
                template_context['mg_info'] = mg_info_details
            else:
                msg = f"Unsupported modality '{report_meta['modality']}' for detailed view of '{report_meta['filename']}'."
                errors_list.append(msg)
    except pydicom.errors.InvalidDicomError:
        msg = f"File '{report_meta['filename']}' is not a valid DICOM format or is corrupted."
        errors_list.append(msg)
    except Exception as e:
        msg = f"An unexpected error occurred while reading report details for '{report_meta['filename']}': {type(e).__name__} - {str(e)}"
        errors_list.append(msg)
        print(f"Detailed error for {report_meta['filename']}: {type(e).__name__} - {e}") 
    if errors_list:
        combined_error_message = "; ".join(errors_list)
        flash(f"Issues loading report: {combined_error_message}", "error")
    template_context['errors'] = errors_list 
    if template_context['modality'] == 'CT' and not template_context['info'] and report_meta:
        template_context['info'] = report_meta 
    if template_context['modality'] == 'DX' and not template_context['dx_info'] and report_meta:
        template_context['dx_info'] = report_meta
    if template_context['modality'] == 'MG' and not template_context['mg_info'] and report_meta:
        template_context['mg_info'] = report_meta
    return render_template('results.html', **template_context)

# --- Comparison and Export Routes ---
def _generate_comparison_plot(x_data, y_data, x_label, y_label, title_prefix, group_name):
    plt.figure(figsize=(max(8, len(x_data) * 0.5 + 2), 6.5))
    plt.bar(x_data, y_data, color='#5eaaa8', width=0.6) 
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'{title_prefix} for {group_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout(pad=1.5)
    safe_group_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', group_name)
    plot_filename_base = title_prefix.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("·","")
    plot_filename = f'{plot_filename_base}_{safe_group_name}_{uuid.uuid4().hex[:6]}.png'
    plot_path = os.path.join(STATIC_FOLDER, plot_filename)
    try:
        plt.savefig(plot_path)
        return url_for('static', filename=plot_filename)
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
        return None
    finally:
        plt.close()

def _create_excel_export(df, sheet_name, download_base_filename):
    try:
        output_stream = BytesIO()
        df.to_excel(output_stream, index=False, sheet_name=sheet_name)
        output_stream.seek(0)
        safe_filename_suffix = re.sub(r'[^a-zA-Z0-9_.-]', '_', sheet_name) 
        download_filename = f'{download_base_filename}_{safe_filename_suffix}.xlsx'
        return send_file(
            output_stream, as_attachment=True, download_name=download_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        flash(f"Error generating Excel file: {e}", "error")
        return None

# CT Comparison Routes (compare_dlp, export_excel_filtered, mean_dlp_comparison, export_mean_dlp_excel, mean_ctdivol_comparison, export_mean_ctdivol_excel)
# These remain unchanged as they are CT-specific.
# ... (Existing CT comparison and export routes from your app.py) ...
@app.route('/compare_dlp')
def compare_dlp():
    if session.get('modality') != 'CT':
        flash("DLP Comparison is for CT modality only.", "warning")
        return redirect(url_for('list_reports'))
    ct_reports = [r for r in REPORT_INDEX if r['modality'] == 'CT']
    if not ct_reports:
        flash("No CT reports uploaded to compare DLP.", "info")
        return render_template('compare_dlp.html', plots=[], tables=[], study_descriptions=[], current_modality='CT')
    records = []
    for report_meta in ct_reports:
        try:
            dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
            if not os.path.exists(dicom_path): continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            _, _, total_dlp_val = extract_ct_dose_info(ds) 
            if total_dlp_val is not None and report_meta.get('study_description', 'N/A') != 'N/A':
                records.append({
                    'Study Description': report_meta.get('study_description', '').strip(),
                    'Patient ID': report_meta.get('Patient ID', 'N/A'),
                    'report_id': report_meta['id'], 'filename': report_meta['filename'],
                    'Total DLP': total_dlp_val, 'study_date': report_meta.get('Study Date', '')
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for DLP comparison: {e}")
    if not records:
        flash("No CT reports with valid DLP data for comparison.", "info")
        return render_template('compare_dlp.html', plots=[], tables=[], study_descriptions=[], current_modality='CT')
    df_all = pd.DataFrame(records)
    plots, tables = [], []
    study_descs_for_page = sorted(list(df_all['Study Description'].unique()))
    selected_filter = request.args.get('study_desc_filter')
    df_to_plot = df_all[df_all['Study Description'] == selected_filter].copy() if selected_filter else df_all.copy()
    if df_to_plot.empty and selected_filter:
        flash(f"No data for Study Description: {selected_filter}", "info")
    for group_name, group_data in df_to_plot.groupby('Study Description'):
        if group_data.empty or group_data['Total DLP'].isnull().all(): continue
        plot_url = _generate_comparison_plot(
            group_data['Patient ID'], group_data['Total DLP'],
            'Patient ID', 'Total DLP (mGy·cm)', 'Total DLP', group_name
        )
        if plot_url: plots.append({'study': group_name, 'plot_url': plot_url})
        tables.append({'study': group_name, 'table': group_data.to_dict(orient='records')})
    return render_template('compare_dlp.html', plots=plots, tables=tables, 
                           study_descriptions=study_descs_for_page, 
                           selected_study=selected_filter, current_modality='CT')

@app.route('/export_excel_filtered')
def export_excel_filtered():
    if session.get('modality') != 'CT':
        flash("Excel export for CT only.", "error")
        return redirect(url_for('list_reports'))
    study_desc_filter = request.args.get('study_desc')
    if not study_desc_filter:
        flash("Select Study Description for export.", "warning")
        return redirect(url_for('compare_dlp'))
    records = []
    for r_meta in filter(lambda r: r['modality'] == 'CT' and r.get('study_description','').strip() == study_desc_filter, REPORT_INDEX):
        try:
            dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], r_meta['save_name'])
            if not os.path.exists(dicom_path): continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            _, _, total_dlp_val = extract_ct_dose_info(ds)
            records.append({
                'Study Description': r_meta.get('study_description', ''), 'Patient ID': r_meta.get('Patient ID', 'N/A'),
                'Filename': r_meta.get('filename', ''), 'Modality': r_meta.get('modality', ''),
                'Total DLP (mGy.cm)': total_dlp_val if total_dlp_val is not None else 'N/A',
                'Study Date': r_meta.get('Study Date', '')
            })
        except Exception as e: print(f"Error for Excel export (CT): {r_meta['filename']} - {e}")
    if not records:
        flash(f"No data for export: {study_desc_filter}", "info")
        return redirect(url_for('compare_dlp'))
    response = _create_excel_export(pd.DataFrame(records), study_desc_filter, "CT_DLP_Export")
    return response if response else redirect(url_for('compare_dlp'))

@app.route('/mean_dlp_comparison')
def mean_dlp_comparison():
    if session.get('modality') != 'CT':
        flash("Mean DLP Comparison is for CT modality only.", "warning")
        return redirect(url_for('list_reports'))
    ct_reports = [r for r in REPORT_INDEX if r['modality'] == 'CT']
    if not ct_reports:
        flash("No CT reports uploaded to calculate Mean DLP.", "info")
        return render_template('mean_dlp_comparison.html',
                               study_data_with_means=[],
                               study_descriptions=[],
                               selected_study=None,
                               current_modality='CT',
                               current_sort_by='study_description',
                               current_sort_order='asc',
                               plot_url_mean_dlp=None)
    records_for_dlp = []
    for report_meta in ct_reports:
        try:
            dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
            if not os.path.exists(dicom_path): continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            _, _, total_dlp_val = extract_ct_dose_info(ds) 
            raw_study_date = get_clean_value(ds, (0x0008,0x0020))
            if report_meta.get('study_description', 'N/A') != 'N/A' and total_dlp_val is not None:
                records_for_dlp.append({
                    'Study Description': report_meta.get('study_description', '').strip(),
                    'Total DLP': total_dlp_val, 
                    'report_count_placeholder': 1, 
                    'Patient ID': report_meta.get('Patient ID', 'N/A'),
                    'study_date': report_meta.get('Study Date', ''),
                    'raw_study_date': raw_study_date if raw_study_date and str(raw_study_date).isdigit() and len(str(raw_study_date)) == 8 else None,
                    'filename': report_meta.get('filename', '')
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for Mean DLP calculation: {e}")
    if not records_for_dlp:
        flash("No CT reports with valid Total DLP data.", "info")
        return render_template('mean_dlp_comparison.html',
                               study_data_with_means=[],
                               study_descriptions=[],
                               selected_study=None,
                               current_modality='CT',
                               current_sort_by='study_description',
                               current_sort_order='asc',
                               plot_url_mean_dlp=None)
    df_dlp = pd.DataFrame(records_for_dlp)
    study_summary_list_dlp = []
    for study_desc, group in df_dlp.groupby('Study Description'):
        mean_dlp_val = group['Total DLP'].mean() if not group['Total DLP'].empty else None
        num_reports = group['report_count_placeholder'].sum()
        study_summary_list_dlp.append({
            'study_description': study_desc,
            'mean_dlp': mean_dlp_val,
            'number_of_reports': num_reports
        })
    sort_by = request.args.get('sort_by', 'study_description')
    sort_order = request.args.get('sort_order', 'asc')
    if sort_order not in ['asc', 'desc']:
        sort_order = 'asc'
    is_reverse = (sort_order == 'desc')
    def sort_key_dlp_summary(item): 
        val = item.get(sort_by)
        if val is None:
            return float('-inf') if sort_order == 'asc' else float('inf')
        if sort_by in ['mean_dlp', 'number_of_reports']: 
            try: return float(val)
            except (ValueError, TypeError): return float('-inf') if sort_order == 'asc' else float('inf')
        if isinstance(val, str):
            return val.lower()
        return val
    try:
        study_summary_list_dlp.sort(key=sort_key_dlp_summary, reverse=is_reverse)
    except TypeError as e:
        flash(f"Could not sort DLP summary data by '{sort_by}'. Error: {e}", "warning")
    all_study_descriptions_dlp = sorted(list(df_dlp['Study Description'].unique()))
    selected_study_filter_dlp = request.args.get('study_desc_filter')
    data_for_page = study_summary_list_dlp
    if selected_study_filter_dlp:
        data_for_page = [s for s in study_summary_list_dlp if s['study_description'] == selected_study_filter_dlp]
    plot_url_mean_dlp = None
    if data_for_page:
        chart_data_points_dlp = [item for item in data_for_page if item.get('mean_dlp') is not None]
        if chart_data_points_dlp:
            study_descs_for_chart_dlp = [item['study_description'] for item in chart_data_points_dlp]
            mean_values_for_chart_dlp = [item['mean_dlp'] for item in chart_data_points_dlp]
            if study_descs_for_chart_dlp and mean_values_for_chart_dlp:
                 plot_url_mean_dlp = _generate_comparison_plot(
                    x_data=study_descs_for_chart_dlp,
                    y_data=mean_values_for_chart_dlp,
                    x_label='Study Description',
                    y_label='Mean Total DLP (mGy·cm)',
                    title_prefix='Mean Total DLP Comparison',
                    group_name=selected_study_filter_dlp if selected_study_filter_dlp else 'All Studies'
                ) 
    return render_template('mean_dlp_comparison.html', 
                           study_data_with_means=data_for_page,
                           study_descriptions=all_study_descriptions_dlp,
                           selected_study=selected_study_filter_dlp,
                           current_modality='CT',
                           current_sort_by=sort_by,
                           current_sort_order=sort_order,
                           plot_url_mean_dlp=plot_url_mean_dlp)

@app.route('/export_mean_dlp_excel')
def export_mean_dlp_excel():
    if session.get('modality') != 'CT':
        flash("Excel export for Mean DLP is for CT modality only.", "error")
        return redirect(url_for('list_reports'))
    ct_reports = [r for r in REPORT_INDEX if r['modality'] == 'CT']
    if not ct_reports:
        flash("No CT reports data to export.", "info")
        return redirect(url_for('mean_dlp_comparison'))
    records_for_dlp_export = []
    for report_meta in ct_reports:
        try:
            dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
            if not os.path.exists(dicom_path): continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            _, _, total_dlp_val = extract_ct_dose_info(ds) 
            if report_meta.get('study_description', 'N/A') != 'N/A' and total_dlp_val is not None:
                records_for_dlp_export.append({
                    'Study Description': report_meta.get('study_description', '').strip(),
                    'Total DLP': total_dlp_val,
                    'report_count_placeholder': 1
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for Excel export (Mean DLP): {e}")
    if not records_for_dlp_export:
        flash("No valid Total DLP data to export.", "info")
        return redirect(url_for('mean_dlp_comparison'))
    df_dlp_export = pd.DataFrame(records_for_dlp_export)
    study_summary_for_excel_dlp = []
    for study_desc, group in df_dlp_export.groupby('Study Description'):
        mean_dlp_val = group['Total DLP'].mean() if not group['Total DLP'].empty else None
        num_reports = group['report_count_placeholder'].sum()
        study_summary_for_excel_dlp.append({
            'Study Description': study_desc,
            'Mean Total DLP (mGy.cm)': round(mean_dlp_val, 2) if mean_dlp_val is not None else 'N/A',
            'Number of Reports': num_reports
        })
    if not study_summary_for_excel_dlp:
        flash("No summary DLP data to export to Excel.", "info")
        return redirect(url_for('mean_dlp_comparison'))
    df_final_export = pd.DataFrame(study_summary_for_excel_dlp)
    study_desc_filter_export_dlp = request.args.get('study_desc_filter')
    if study_desc_filter_export_dlp:
        df_final_export = df_final_export[df_final_export['Study Description'] == study_desc_filter_export_dlp]
        if df_final_export.empty:
            flash(f"No data to export for filtered Study Description: {study_desc_filter_export_dlp}", "info")
            return redirect(url_for('mean_dlp_comparison', study_desc_filter=study_desc_filter_export_dlp))
        sheet_name = f"MeanDLP_{study_desc_filter_export_dlp[:20]}"
    else:
        sheet_name = 'Mean DLP Summary'
    response = _create_excel_export(df_final_export, sheet_name, "Mean_DLP_Summary_Export") 
    return response if response else redirect(url_for('mean_dlp_comparison'))

@app.route('/mean_ctdivol_comparison')
def mean_ctdivol_comparison():
    if session.get('modality') != 'CT':
        flash("Mean CTDIvol Comparison is for CT modality only.", "warning")
        return redirect(url_for('list_reports'))
    ct_reports = [r for r in REPORT_INDEX if r['modality'] == 'CT']
    if not ct_reports: 
        flash("No CT reports uploaded to calculate Mean CTDIvol.", "info")
        return render_template('mean_ctdivol_comparison.html',
                               study_data_with_means=[],
                               study_descriptions=[],
                               selected_study=None,
                               current_modality='CT',
                               current_sort_by='study_description',
                               current_sort_order='asc',
                               plot_url_mean_ctdivol=None) 
    records_for_ctdivol = []
    for report_meta in ct_reports:
        try:
            dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
            if not os.path.exists(dicom_path): continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            _, ct_series_data, _ = extract_ct_dose_info(ds) 
            ctdivol_values_for_report = []
            if ct_series_data:
                for event_item in ct_series_data:
                    if event_item.get('Type') != 'Scout/Localizer' and \
                       event_item.get('CTDIvol') not in [None, '', 'N/A']:
                        try:
                            ctdivol_values_for_report.append(float(event_item['CTDIvol']))
                        except (ValueError, TypeError):
                            pass
            if report_meta.get('study_description', 'N/A') != 'N/A':
                records_for_ctdivol.append({
                    'Study Description': report_meta.get('study_description', '').strip(),
                    'ctdivol_events': ctdivol_values_for_report,
                    'report_count_placeholder': 1 
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for Mean CTDIvol calculation: {e}")
    if not records_for_ctdivol:
        flash("No CT reports with valid CTDIvol data.", "info")
        return render_template('mean_ctdivol_comparison.html',
                               study_data_with_means=[],
                               study_descriptions=[],
                               selected_study=None,
                               current_modality='CT',
                               current_sort_by='study_description',
                               current_sort_order='asc',
                               plot_url_mean_ctdivol=None) 
    df_ctdivol = pd.DataFrame(records_for_ctdivol)
    study_summary_list = []
    for study_desc, group in df_ctdivol.groupby('Study Description'):
        all_events_for_study = []
        for event_list in group['ctdivol_events']:
            all_events_for_study.extend(event_list)
        mean_val = None
        if all_events_for_study:
            mean_val = sum(all_events_for_study) / len(all_events_for_study)
        num_reports = group['report_count_placeholder'].sum()
        study_summary_list.append({
            'study_description': study_desc,
            'mean_ctdivol': mean_val,
            'number_of_reports': num_reports,
            'total_scan_events': len(all_events_for_study)
        })
    sort_by = request.args.get('sort_by', 'study_description')
    sort_order = request.args.get('sort_order', 'asc')
    if sort_order not in ['asc', 'desc']: sort_order = 'asc'
    is_reverse = (sort_order == 'desc')
    def sort_key_summary(item): 
        val = item.get(sort_by)
        if val is None:
            return float('-inf') if sort_order == 'asc' else float('inf')
        if sort_by in ['mean_ctdivol', 'number_of_reports', 'total_scan_events']:
            try: return float(val)
            except (ValueError, TypeError): return float('-inf') if sort_order == 'asc' else float('inf')
        if isinstance(val, str):
            return val.lower()
        return val
    try:
        study_summary_list.sort(key=sort_key_summary, reverse=is_reverse)
    except TypeError as e:
        flash(f"Could not sort summary data by '{sort_by}'. Error: {e}", "warning")
    all_study_descriptions = sorted(list(df_ctdivol['Study Description'].unique()))
    selected_study_filter = request.args.get('study_desc_filter')
    data_for_chart_and_table = study_summary_list
    if selected_study_filter:
        data_for_chart_and_table = [s for s in study_summary_list if s['study_description'] == selected_study_filter]
    plot_url_mean_ctdivol = None
    if data_for_chart_and_table:
        chart_data_points = [item for item in data_for_chart_and_table if item.get('mean_ctdivol') is not None]
        if chart_data_points:
            study_descs_for_chart = [item['study_description'] for item in chart_data_points]
            mean_values_for_chart = [item['mean_ctdivol'] for item in chart_data_points]
            if study_descs_for_chart and mean_values_for_chart: 
                 plot_url_mean_ctdivol = _generate_comparison_plot(
                    x_data=study_descs_for_chart,
                    y_data=mean_values_for_chart,
                    x_label='Study Description',
                    y_label='Mean CTDIvol (mGy per scan event)',
                    title_prefix='Mean CTDIvol Comparison',
                    group_name=selected_study_filter if selected_study_filter else 'All Studies'
                ) 
    return render_template('mean_ctdivol_comparison.html',
                           study_data_with_means=data_for_chart_and_table, 
                           study_descriptions=all_study_descriptions,
                           selected_study=selected_study_filter,
                           current_modality='CT',
                           current_sort_by=sort_by,
                           current_sort_order=sort_order,
                           plot_url_mean_ctdivol=plot_url_mean_ctdivol) 

@app.route('/export_mean_ctdivol_excel')
def export_mean_ctdivol_excel():
    if session.get('modality') != 'CT':
        flash("Excel export for Mean CTDIvol is for CT modality only.", "error")
        return redirect(url_for('list_reports'))
    ct_reports = [r for r in REPORT_INDEX if r['modality'] == 'CT']
    if not ct_reports:
        flash("No CT reports data to export.", "info")
        return redirect(url_for('mean_ctdivol_comparison'))
    records_for_ctdivol = []
    for report_meta in ct_reports:
        try:
            dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
            if not os.path.exists(dicom_path): continue
            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            _, ct_series_data, _ = extract_ct_dose_info(ds) 
            ctdivol_values_for_report = []
            if ct_series_data:
                for event_item in ct_series_data:
                    if event_item.get('Type') != 'Scout/Localizer' and \
                       event_item.get('CTDIvol') not in [None, '', 'N/A']:
                        try:
                            ctdivol_values_for_report.append(float(event_item['CTDIvol']))
                        except (ValueError, TypeError):
                            pass
            if report_meta.get('study_description', 'N/A') != 'N/A':
                records_for_ctdivol.append({
                    'Study Description': report_meta.get('study_description', '').strip(),
                    'ctdivol_events': ctdivol_values_for_report,
                    'report_count_placeholder': 1
                })
        except Exception as e:
            print(f"Error processing '{report_meta['filename']}' for Excel export (Mean CTDIvol): {e}")
    if not records_for_ctdivol:
        flash("No valid CTDIvol data to export.", "info")
        return redirect(url_for('mean_ctdivol_comparison'))
    df_ctdivol = pd.DataFrame(records_for_ctdivol)
    study_summary_for_excel = []
    for study_desc, group in df_ctdivol.groupby('Study Description'):
        all_events_for_study = []
        for event_list in group['ctdivol_events']:
            all_events_for_study.extend(event_list)
        mean_val = None
        if all_events_for_study:
            mean_val = sum(all_events_for_study) / len(all_events_for_study)
        num_reports = group['report_count_placeholder'].sum()
        study_summary_for_excel.append({
            'Study Description': study_desc,
            'Mean CTDIvol (mGy per scan event)': round(mean_val, 2) if mean_val is not None else 'N/A',
            'Number of Reports': num_reports,
            'Total Scan Events Used': len(all_events_for_study)
        })
    if not study_summary_for_excel:
        flash("No summary data to export to Excel.", "info")
        return redirect(url_for('mean_ctdivol_comparison'))
    df_export = pd.DataFrame(study_summary_for_excel)
    study_desc_filter_export = request.args.get('study_desc_filter')
    if study_desc_filter_export:
        df_export = df_export[df_export['Study Description'] == study_desc_filter_export]
        if df_export.empty:
            flash(f"No data to export for filtered Study Description: {study_desc_filter_export}", "info")
            return redirect(url_for('mean_ctdivol_comparison', study_desc_filter=study_desc_filter_export))
        sheet_name = study_desc_filter_export[:30] 
    else:
        sheet_name = 'Mean CTDIvol Summary'
    response = _create_excel_export(df_export, sheet_name, "Mean_CTDIvol_Summary_Export") 
    return response if response else redirect(url_for('mean_ctdivol_comparison'))

# DX Comparison Route (compare_dap, export_excel_dx_dap_filtered)
# These remain unchanged as they are DX-specific.
# ... (Existing DX comparison and export routes from your app.py) ...
@app.route('/compare_dap')
def compare_dap():
    if session.get('modality') != 'DX':
        flash("DAP Comparison is for DX modality only.", "warning")
        return redirect(url_for('list_reports'))
    dx_reports = [r for r in REPORT_INDEX if r['modality'] == 'DX']
    if not dx_reports:
        flash("No DX reports uploaded to compare DAP.", "info")
        return render_template('compare_dap.html', plots=[], tables=[], body_parts_examined=[], current_modality='DX')
    records = []
    for r_meta in dx_reports:
        dap_val_str = r_meta.get(TARGET_DAP_STORAGE_KEY_DX, 'N/A')
        dap_val_num = None
        if not str(dap_val_str).lower().startswith('n/a'):
            try: dap_val_num = float(dap_val_str)
            except ValueError: print(f"Could not parse DAP '{dap_val_str}' for {r_meta['filename']}")
        if dap_val_num is not None and r_meta.get('body_part') and r_meta.get('body_part','N/A') != 'N/A':
            records.append({
                'Body Part Examined': r_meta.get('body_part','').strip().upper(), 
                'Patient ID': r_meta.get('Patient ID', 'N/A'), 'report_id': r_meta['id'],
                'filename': r_meta['filename'], TARGET_DAP_UNIT_LABEL_DX: dap_val_num,
                'study_date': r_meta.get('Study Date', '')
            })
    if not records:
        flash("No DX reports with valid DAP and Body Part for comparison.", "info")
        return render_template('compare_dap.html', plots=[], tables=[], body_parts_examined=[], current_modality='DX')
    df_all = pd.DataFrame(records)
    df_all[TARGET_DAP_UNIT_LABEL_DX] = pd.to_numeric(df_all[TARGET_DAP_UNIT_LABEL_DX], errors='coerce')
    df_all.dropna(subset=[TARGET_DAP_UNIT_LABEL_DX], inplace=True)
    if df_all.empty:
        flash("No numeric DAP data after filtering.", "info")
        return render_template('compare_dap.html', plots=[], tables=[], body_parts_examined=[], current_modality='DX')
    plots, tables = [], []
    body_parts_for_page = sorted(list(df_all['Body Part Examined'].unique()))
    selected_filter = request.args.get('body_part_filter')
    df_to_process = df_all[df_all['Body Part Examined'] == selected_filter].copy() if selected_filter else df_all.copy()
    if df_to_process.empty and selected_filter: flash(f"No data for Body Part: {selected_filter}", "info")
    for group_name, group_exams in df_to_process.groupby('Body Part Examined'):
        if group_exams.empty or group_exams[TARGET_DAP_UNIT_LABEL_DX].isnull().all(): continue
        agg_dap = group_exams.groupby('Patient ID')[TARGET_DAP_UNIT_LABEL_DX].mean().reset_index()
        agg_label = "Average"
        if agg_dap.empty: continue
        plot_url = _generate_comparison_plot(
            agg_dap['Patient ID'], agg_dap[TARGET_DAP_UNIT_LABEL_DX],
            'Patient ID', f'{agg_label} {TARGET_DAP_UNIT_LABEL_DX}', f'{agg_label} DAP', group_name
        )
        if plot_url: plots.append({'body_part': group_name, 'plot_url': plot_url, 'aggregation_method': agg_label})
        table_detail = group_exams.sort_values(by=['Patient ID', 'study_date'])
        tables.append({'body_part': group_name, 'table': table_detail.to_dict(orient='records')})
    return render_template('compare_dap.html', plots=plots, tables=tables, 
                           body_parts_examined=body_parts_for_page, 
                           selected_body_part=selected_filter, current_modality='DX')

@app.route('/export_excel_dx_dap_filtered')
def export_excel_dx_dap_filtered():
    if session.get('modality') != 'DX':
        flash("Excel export for DX only.", "error")
        return redirect(url_for('list_reports'))
    body_part_filter = request.args.get('body_part')
    if not body_part_filter:
        flash("Select Body Part for export.", "warning")
        return redirect(url_for('compare_dap'))
    records = []
    for r_meta in filter(lambda r: r['modality'] == 'DX' and r.get('body_part','').strip().upper() == body_part_filter.upper(), REPORT_INDEX):
        dap_val_str = r_meta.get(TARGET_DAP_STORAGE_KEY_DX, 'N/A')
        dap_val_num = None
        if not str(dap_val_str).lower().startswith('n/a'):
            try: dap_val_num = float(dap_val_str)
            except ValueError: pass
        records.append({
            'Body Part Examined': r_meta.get('body_part', ''), 'Patient ID': r_meta.get('Patient ID', 'N/A'),
            'Filename': r_meta.get('filename', ''), 'Modality': r_meta.get('modality', ''),
            TARGET_DAP_UNIT_LABEL_DX: dap_val_num if dap_val_num is not None else 'N/A',
            'Study Date': r_meta.get('Study Date', '')
        })
    if not records:
        flash(f"No data for export: {body_part_filter}", "info")
        return redirect(url_for('compare_dap'))
    response = _create_excel_export(pd.DataFrame(records), body_part_filter, "DX_DAP_Export")
    return response if response else redirect(url_for('compare_dap'))

@app.route('/compare_mg_organ_dose')
def compare_mg_organ_dose():
    if session.get('modality') != 'MG':
        flash("Organ Dose Comparison is for MG modality only.", "warning")
        return redirect(url_for('list_reports'))

    mg_reports = [r for r in REPORT_INDEX if r['modality'] == 'MG']
    if not mg_reports:
        flash("No MG reports uploaded to compare Organ Dose.", "info")
        return render_template('compare_organ_dose_mg.html',
                               summary_data=[],
                               plot_url_mg_organ_dose=None,
                               current_modality='MG',
                               # เพิ่มตัวแปรสำหรับ filter (ถ้ามี)
                               body_parts_examined = sorted(list(set(r.get('body_part', 'N/A') for r in mg_reports if r.get('body_part', 'N/A') != 'N/A'))),
                               view_positions = sorted(list(set(r.get('view_position', 'N/A') for r in mg_reports if r.get('view_position', 'N/A') != 'N/A'))),
                               selected_body_part_filter = request.args.get('body_part_filter'),
                               selected_view_position_filter = request.args.get('view_position_filter')
                               )

    records = []
    for r_meta in mg_reports:
        # ใช้ค่าที่เก็บไว้ใน REPORT_INDEX โดยตรง
        organ_dose_str = r_meta.get(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'N/A')
        organ_dose_val = None
        if not str(organ_dose_str).lower().startswith('n/a') and "invalid" not in str(organ_dose_str).lower():
            try:
                organ_dose_val = float(organ_dose_str)
            except ValueError:
                print(f"Could not parse Organ Dose '{organ_dose_str}' for report ID {r_meta['id']} in comparison.")
        
        if organ_dose_val is not None:
            records.append({
                'Patient ID': r_meta.get('Patient ID', 'N/A'),
                'Study Date': r_meta.get('Study Date', 'N/A'), # วันที่ที่ format แล้ว
                'Raw Study Date': r_meta.get('Raw Study Date', ''), # YYYYMMDD สำหรับการ sort ที่แม่นยำ
                TARGET_ORGAN_DOSE_STORAGE_KEY_MG: organ_dose_val,
                'Body Part Examined': r_meta.get('body_part', 'N/A'),
                'View Position': r_meta.get('view_position', 'N/A'),
                'report_id': r_meta.get('id') # อาจมีประโยชน์สำหรับการ link กลับไปดู report
            })

    if not records:
        flash("No MG reports with valid Organ Dose data found for comparison.", "info")
        return render_template('compare_organ_dose_mg.html',
                               summary_data=[],
                               plot_url_mg_organ_dose=None,
                               current_modality='MG',
                               body_parts_examined = sorted(list(set(r.get('body_part', 'N/A') for r in mg_reports if r.get('body_part', 'N/A') != 'N/A'))),
                               view_positions = sorted(list(set(r.get('view_position', 'N/A') for r in mg_reports if r.get('view_position', 'N/A') != 'N/A'))),
                               selected_body_part_filter = request.args.get('body_part_filter'),
                               selected_view_position_filter = request.args.get('view_position_filter')
                               )

    df_all_mg = pd.DataFrame(records)

    # --- Apply Filters ---
    body_part_filter = request.args.get('body_part_filter')
    view_position_filter = request.args.get('view_position_filter')

    filtered_df = df_all_mg.copy()
    filter_active_message = []

    if body_part_filter:
        filtered_df = filtered_df[filtered_df['Body Part Examined'] == body_part_filter]
        filter_active_message.append(f"Body Part: {body_part_filter}")
    if view_position_filter:
        filtered_df = filtered_df[filtered_df['View Position'] == view_position_filter]
        filter_active_message.append(f"View: {view_position_filter}")
    
    if filtered_df.empty:
        flash(f"No MG data found matching the filter criteria: {', '.join(filter_active_message) if filter_active_message else 'None'}.", "info")
        # ยังคงส่งค่า filter ที่เลือกไปให้ template
        return render_template('compare_organ_dose_mg.html',
                               summary_data=[],
                               plot_url_mg_organ_dose=None,
                               current_modality='MG',
                               body_parts_examined = sorted(list(set(df_all_mg['Body Part Examined'].unique()))),
                               view_positions = sorted(list(set(df_all_mg['View Position'].unique()))),
                               selected_body_part_filter = body_part_filter,
                               selected_view_position_filter = view_position_filter
                               )


    # Group by Patient ID and Study Date, then calculate mean and count
    # Make sure to sort by Raw Study Date before grouping if that influences the average
    # However, for averaging per day, sorting before grouping by date isn't strictly necessary
    # if the date itself is part of the group key.
    
    # Sort by Raw Study Date first to ensure chronological order if needed for display later
    # For groupby, the order of input doesn't affect the groups themselves.
    # filtered_df_sorted = filtered_df.sort_values(by=['Patient ID', 'Raw Study Date'])


    patient_daily_avg_dose = filtered_df.groupby(['Patient ID', 'Study Date']).agg(
        average_organ_dose=(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'mean'),
        number_of_studies=('report_id', 'count') #นับจำนวน report ที่ถูกนำมาเฉลี่ย
    ).reset_index()

    # Sort the final summary data, e.g., by Patient ID then Study Date
    patient_daily_avg_dose = patient_daily_avg_dose.sort_values(by=['Patient ID', 'Study Date'])


    plot_url_mg_organ_dose = None
    summary_list_for_template = []

    if not patient_daily_avg_dose.empty:
        summary_list_for_template = patient_daily_avg_dose.to_dict(orient='records')
        
        # Prepare data for plotting
        # Ensure there are enough distinct data points to make a meaningful plot
        if len(patient_daily_avg_dose) > 1 : # Only plot if more than one data point
            x_labels_plot = [f"{row['Patient ID']} ({row['Study Date']})" for _, row in patient_daily_avg_dose.iterrows()]
            y_values_plot = patient_daily_avg_dose['average_organ_dose'].tolist()

            plot_title_suffix = "All Studies"
            if filter_active_message:
                plot_title_suffix = ", ".join(filter_active_message)


            plot_url_mg_organ_dose = _generate_comparison_plot(
                x_data=x_labels_plot,
                y_data=y_values_plot,
                x_label='Patient (Study Date)',
                y_label=f'Average {TARGET_ORGAN_DOSE_UNIT_LABEL_MG}',
                title_prefix=f'Average Organ Dose (MG) - {plot_title_suffix}',
                group_name="PatientAverages" # Generic group name for the plot file
            )
        elif len(patient_daily_avg_dose) == 1:
             flash("Only one data point after aggregation; plot not generated.", "info")


    # For filter dropdowns
    all_body_parts = sorted(list(set(df_all_mg['Body Part Examined'].unique())))
    all_view_positions = sorted(list(set(df_all_mg['View Position'].unique())))


    return render_template('compare_organ_dose_mg.html',
                           summary_data=summary_list_for_template,
                           plot_url_mg_organ_dose=plot_url_mg_organ_dose,
                           current_modality='MG',
                           TARGET_ORGAN_DOSE_UNIT_LABEL_MG=TARGET_ORGAN_DOSE_UNIT_LABEL_MG,
                           body_parts_examined=all_body_parts, # Pass all available for dropdown
                           view_positions=all_view_positions, # Pass all available for dropdown
                           selected_body_part_filter=body_part_filter,
                           selected_view_position_filter=view_position_filter,
                           filter_active_message = ", ".join(filter_active_message) if filter_active_message else None
                           )


@app.route('/export_excel_mg_organ_dose_avg')
def export_excel_mg_organ_dose_avg():
    if session.get('modality') != 'MG':
        flash("Excel export for Average MG Organ Dose is for MG modality only.", "error")
        return redirect(url_for('list_reports'))

    # Re-fetch and process data similar to the compare_mg_organ_dose route
    mg_reports = [r for r in REPORT_INDEX if r['modality'] == 'MG']
    if not mg_reports:
        flash("No MG reports data to export.", "info")
        return redirect(url_for('compare_mg_organ_dose'))

    records = []
    for r_meta in mg_reports:
        organ_dose_str = r_meta.get(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'N/A')
        organ_dose_val = None
        if not str(organ_dose_str).lower().startswith('n/a') and "invalid" not in str(organ_dose_str).lower():
            try:
                organ_dose_val = float(organ_dose_str)
            except ValueError:
                pass # Skip if parsing fails
        
        if organ_dose_val is not None:
            records.append({
                'Patient ID': r_meta.get('Patient ID', 'N/A'),
                'Study Date': r_meta.get('Study Date', 'N/A'),
                TARGET_ORGAN_DOSE_STORAGE_KEY_MG: organ_dose_val,
                'Body Part Examined': r_meta.get('body_part', 'N/A'),
                'View Position': r_meta.get('view_position', 'N/A'),
                'Report ID': r_meta.get('id')
            })
    
    if not records:
        flash("No valid MG Organ Dose data to export.", "info")
        return redirect(url_for('compare_mg_organ_dose'))

    df_all_mg_export = pd.DataFrame(records)

    # Apply filters if they are passed as query parameters
    body_part_filter = request.args.get('body_part_filter')
    view_position_filter = request.args.get('view_position_filter')
    
    filtered_df_export = df_all_mg_export.copy()
    sheet_name_suffix = "All"

    if body_part_filter:
        filtered_df_export = filtered_df_export[filtered_df_export['Body Part Examined'] == body_part_filter]
        sheet_name_suffix = body_part_filter.replace(" ", "_")[:20] 
    if view_position_filter:
        filtered_df_export = filtered_df_export[filtered_df_export['View Position'] == view_position_filter]
        if body_part_filter: # Append if both filters active
             sheet_name_suffix += f"_{view_position_filter.replace(' ', '_')[:10]}"
        else: # Only view position filter active
             sheet_name_suffix = view_position_filter.replace(" ", "_")[:20]


    if filtered_df_export.empty:
        flash(f"No MG data found to export matching the filter criteria.", "info")
        return redirect(url_for('compare_mg_organ_dose', 
                                body_part_filter=body_part_filter, 
                                view_position_filter=view_position_filter))


    patient_daily_avg_dose_export = filtered_df_export.groupby(['Patient ID', 'Study Date']).agg(
        average_organ_dose=(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'mean'),
        number_of_studies=('Report ID', 'count')
    ).reset_index()

    patient_daily_avg_dose_export.rename(columns={
        'average_organ_dose': f'Average {TARGET_ORGAN_DOSE_UNIT_LABEL_MG}',
        'number_of_studies': 'Number of Studies Averaged'
    }, inplace=True)
    
    patient_daily_avg_dose_export = patient_daily_avg_dose_export.sort_values(by=['Patient ID', 'Study Date'])


    if patient_daily_avg_dose_export.empty:
        flash("No summary MG Organ Dose data to export to Excel.", "info")
        return redirect(url_for('compare_mg_organ_dose', 
                                body_part_filter=body_part_filter, 
                                view_position_filter=view_position_filter))

    response = _create_excel_export(patient_daily_avg_dose_export, 
                                    f"AvgOrganDose_{sheet_name_suffix}", 
                                    "MG_Avg_Organ_Dose_Export")
    return response if response else redirect(url_for('compare_mg_organ_dose', 
                                                      body_part_filter=body_part_filter, 
                                                      view_position_filter=view_position_filter))


# Patient Specific Routes
@app.route('/search_patient', methods=['GET', 'POST'])
def search_patient():
    current_modality = session.get('modality')
    if not current_modality:
        flash("Select modality before searching.", "warning")
        return redirect(url_for('select_modality'))
    results, patient_id_search = [], ''
    if request.method == 'POST':
        patient_id_search = request.form.get('patient_id', '').strip()
        if patient_id_search:
            for r_meta in filter(lambda r: r.get('Patient ID','').strip() == patient_id_search and r['modality'] == current_modality, REPORT_INDEX):
                entry = {'report_id': r_meta['id'], 'study_date': r_meta.get('Study Date', 'N/A'), 'filename': r_meta['filename']}
                if current_modality == 'CT':
                    entry['study_description'] = r_meta.get('study_description', 'N/A')
                    entry['total_dlp'] = 'N/A' 
                    try: 
                        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], r_meta['save_name'])
                        if os.path.exists(dicom_path):
                            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                            _, _, total_dlp_val = extract_ct_dose_info(ds)
                            if total_dlp_val is not None: entry['total_dlp'] = total_dlp_val
                    except Exception as e: print(f"Error getting TotalDLP for search: {e}")
                elif current_modality == 'DX':
                    entry['body_part'] = r_meta.get('body_part', 'N/A')
                    entry['dap_value_label'] = TARGET_DAP_UNIT_LABEL_DX 
                    entry['dap_value'] = r_meta.get(TARGET_DAP_STORAGE_KEY_DX, 'N/A')
                elif current_modality == 'MG':
                    entry['study_description'] = r_meta.get('study_description', 'N/A') 
                    entry['organ_dose_label'] = TARGET_ORGAN_DOSE_UNIT_LABEL_MG
                    entry['organ_dose_value'] = r_meta.get(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'N/A')
                results.append(entry)
            if not results: flash(f"No {current_modality} reports for Patient ID '{patient_id_search}'.", "info")
        elif request.method == 'POST': flash("Enter Patient ID to search.", "warning")
    return render_template('search_patient.html', results=results, 
                           patient_id=patient_id_search, current_modality=current_modality)

@app.route('/compare_patient/<patient_id>')
def compare_patient(patient_id):
    current_modality = session.get('modality')
    if not current_modality:
        flash("Select modality first.", "error")
        return redirect(url_for('select_modality'))
    patient_id_stripped = patient_id.strip()
    reports_for_patient = [r for r in REPORT_INDEX if r.get('Patient ID','').strip() == patient_id_stripped and r['modality'] == current_modality]
    if not reports_for_patient:
        flash(f"No {current_modality} reports for Patient ID '{patient_id_stripped}'.", "info")
        return render_template('compare_patient.html', patient_id=patient_id_stripped, studies=[], current_modality=current_modality)
    studies_data, plot_url, value_axis_label = [], None, ""
    for r_meta in reports_for_patient:
        entry = {'report_id': r_meta['id'], 'study_date': r_meta.get('Study Date','N/A'), 'filename': r_meta['filename']}
        numeric_val = None
        if current_modality == 'CT':
            entry['study_description'] = r_meta.get('study_description', 'N/A')
            value_axis_label = 'Total DLP (mGy·cm)'
            entry['group_label'] = f"{entry['study_description'] or 'N/A'} ({entry['study_date']})"
            try: 
                dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], r_meta['save_name'])
                if os.path.exists(dicom_path):
                    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                    _, _, total_dlp_val = extract_ct_dose_info(ds)
                    if total_dlp_val is not None: numeric_val = total_dlp_val
            except Exception as e: print(f"Error for patient DLP plot: {e}")
        elif current_modality == 'DX':
            entry['body_part'] = r_meta.get('body_part', 'N/A')
            value_axis_label = TARGET_DAP_UNIT_LABEL_DX
            entry['group_label'] = f"{entry['body_part'] or 'N/A'} ({entry['study_date']})"
            dap_str = r_meta.get(TARGET_DAP_STORAGE_KEY_DX, 'N/A')
            if not str(dap_str).lower().startswith('n/a') and "invalid" not in str(dap_str).lower():
                try: numeric_val = float(dap_str)
                except ValueError: pass
        elif current_modality == 'MG':
            entry['study_description'] = r_meta.get('study_description', 'N/A') 
            value_axis_label = TARGET_ORGAN_DOSE_UNIT_LABEL_MG
            entry['group_label'] = f"{entry.get('study_description', 'N/A')} ({entry['study_date']})"
            organ_dose_str = r_meta.get(TARGET_ORGAN_DOSE_STORAGE_KEY_MG, 'N/A')
            if not str(organ_dose_str).lower().startswith('n/a') and "invalid" not in str(organ_dose_str).lower():
                try: numeric_val = float(organ_dose_str)
                except ValueError: pass
        entry['value'] = numeric_val
        studies_data.append(entry) 
    plot_data_points = [s for s in studies_data if s.get('value') is not None]
    if plot_data_points:
        plot_url = _generate_comparison_plot(
            [s['group_label'] for s in plot_data_points], [s['value'] for s in plot_data_points],
            'Study Details (Date)', value_axis_label, value_axis_label, patient_id_stripped
        )
        if not plot_url: flash("Error generating patient comparison plot.", "error")
    else:
        flash(f"No numeric data to plot for Patient '{patient_id_stripped}'.", "warning")
    return render_template('compare_patient.html', patient_id=patient_id_stripped, plot_url=plot_url, 
                           studies=studies_data, current_modality=current_modality, 
                           value_label=value_axis_label or "Dose Value")

# ==============================================================================
# ==== MAIN EXECUTION ====
# ==============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
