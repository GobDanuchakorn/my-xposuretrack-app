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
from pydicom.tag import Tag # Not strictly necessary if using hex codes directly often
import datetime
from io import BytesIO # For in-memory Excel file generation

# ==============================================================================
# ==== CONFIGURATION & APP INITIALIZATION ====
# ==============================================================================

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'.dcm'}
PER_PAGE = 20 # For pagination

# Target DAP unit and keys for DX modality
TARGET_DAP_UNIT_LABEL_DX = 'DAP (mGy·cm²)'
TARGET_DAP_STORAGE_KEY_DX = 'dap_value_mgy_cm2' # Key for storing converted DAP in REPORT_INDEX
DAP_CONVERSION_FACTOR_FROM_DGY_CM2_TO_MGY_CM2 = 100 # 1 dGy.cm^2 = 100 mGy.cm^2

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-please-change') # Ensure this is strong and from env in prod
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100000 * 1024 * 1024  

# Make zip available in Jinja2 templates if needed (though not used in current examples)
app.jinja_env.globals.update(zip=zip)

# Ensure essential folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'css'), exist_ok=True) # If you plan to use static/css/main.css

# In-memory report index (Replace with a database for production)
REPORT_INDEX = []

@app.context_processor
def inject_current_year():
    """Injects the current year into all templates."""
    return {'current_year': datetime.datetime.now().year}

# ==============================================================================
# ==== UTILITY FUNCTIONS ====
# ==============================================================================

def allowed_file(filename):
    """Checks if the uploaded file has an allowed DICOM extension."""
    return '.' in filename and \
           os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder_path, extensions_to_delete=None):
    """
    Deletes all files in a specified folder.
    Optionally, only deletes files with certain extensions.
    """
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
                        shutil.rmtree(file_path) # Recursively delete directory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    except Exception as e:
        print(f"Error listing directory {folder_path} for clearing: {e}")


def format_date(dcm_date_str):
    """Converts DICOM date string (YYYYMMDD) to ISO 8601 format (YYYY-MM-DD)."""
    if isinstance(dcm_date_str, str) and len(dcm_date_str) == 8 and dcm_date_str.isdigit():
        try:
            return f"{dcm_date_str[:4]}-{dcm_date_str[4:6]}-{dcm_date_str[6:8]}"
        except ValueError: # Should not happen if isdigit() and len == 8 passed
            return dcm_date_str 
    return str(dcm_date_str) # Return original or stringified version if not typical format

def get_clean_value(dataset, tag_or_keyword, default='N/A'):
    """Safely extracts, cleans, and returns a DICOM tag's value as a string."""
    value_to_process = None
    try:
        raw_element_or_value = dataset.get(tag_or_keyword, None)

        if raw_element_or_value is None:
            return default

        if isinstance(raw_element_or_value, pydicom.dataelem.DataElement):
            if raw_element_or_value.value is None:
                return default
            if hasattr(raw_element_or_value, 'VM') and raw_element_or_value.VM == 0:
                return default # Explicitly empty tag
            value_to_process = raw_element_or_value.value
        else:
            value_to_process = raw_element_or_value # Assumed to be the value directly
        
        if value_to_process is None:
             return default

        if isinstance(value_to_process, pydicom.multival.MultiValue):
            return ', '.join(map(str, value_to_process)).strip()
        elif isinstance(value_to_process, (bytes, bytearray)):
            try: # Attempt to decode common encodings
                return value_to_process.decode('utf-8', errors='replace').strip()
            except UnicodeDecodeError:
                try: return value_to_process.decode('latin-1', errors='replace').strip()
                except: return "Binary Data (Undecodable)"
        return str(value_to_process).strip()
    except Exception as e:
        # Log detailed error for debugging
        print(f"Error getting value for tag/keyword '{tag_or_keyword}': {type(e).__name__} - {e}")
        return default

# ==============================================================================
# ==== DICOM DOSE REPORT EXTRACTION FUNCTIONS ====
# ==============================================================================

def extract_dx_dose_info(ds):
    """Extracts and processes dose-related information for DX modality."""
    info = {}
    info['Patient ID'] = get_clean_value(ds, (0x0010, 0x0020))
    info['Study Date'] = format_date(get_clean_value(ds, (0x0008, 0x0020)))

    # Normalize Body Part Examined
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
            except ValueError: # If float conversion fails for non-"N/A" string
                info[display_name] = f"Invalid: {value_str}"
        else:
            info[display_name] = value_str
            
    # DAP Extraction and Conversion
    dap_val_str_dgycm2 = get_clean_value(ds, (0x0018, 0x115E), 'N/A')
    dap_tag_id_str = '(0018,115E)'
    info['DAP (dGy·cm²)'] = f'N/A {dap_tag_id_str}' # Default for original unit
    info[TARGET_DAP_UNIT_LABEL_DX] = f'N/A {dap_tag_id_str}' # Default for target unit

    if not dap_val_str_dgycm2.lower().startswith('n/a'):
        try:
            dap_dgycm2 = float(dap_val_str_dgycm2)
            info['DAP (dGy·cm²)'] = f"{dap_dgycm2:.2f}"
            info[TARGET_DAP_UNIT_LABEL_DX] = f"{(dap_dgycm2 * DAP_CONVERSION_FACTOR_FROM_DGY_CM2_TO_MGY_CM2):.2f}"
        except ValueError:
            info['DAP (dGy·cm²)'] = f"Invalid {dap_tag_id_str}: {dap_val_str_dgycm2}"
            info[TARGET_DAP_UNIT_LABEL_DX] = f"Invalid {dap_tag_id_str}: {dap_val_str_dgycm2}"

    # Entrance Dose Extraction
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
    """Validates key DX parameters against typical clinical ranges."""
    validations = {}
    sid_str = info.get('SID (mm)', 'N/A')
    if not sid_str.lower().startswith('n/a') and not sid_str.lower().startswith('invalid'):
        try:
            sid = float(sid_str)
            validations['SID Status'] = 'Valid' if 700 <= sid <= 1800 else f'Outlier ({sid}mm)'
        except ValueError: # Should have been caught by previous try-except, but as a safeguard
            validations['SID Status'] = 'Invalid SID value'
    else:
        validations['SID Status'] = sid_str # Propagate N/A or Invalid status

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
    """Extracts and processes dose-related information for CT modality."""
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
                total_dlp = None # Ensure it's None if parsing fails

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

            # --- ส่วนที่เพิ่มเข้ามา ---
            kvp_val = getattr(item, 'KVP', None)
            kvp_display = ''
            if kvp_val is not None:
                try: kvp_display = int(round(float(kvp_val)))
                except (ValueError, TypeError): kvp_display = str(kvp_val)

            # --- ส่วนที่ปรับปรุงสำหรับการอ่าน XRayTubeCurrentInuA และแปลงเป็น mA ---
            tube_current_in_ua_val = getattr(item, 'XRayTubeCurrentInuA', None) # (0018,8151)
            tube_current_display_ma = '' # เปลี่ยนชื่อตัวแปรเพื่อความชัดเจน
            if tube_current_in_ua_val is not None:
                try:
                    # แปลงจาก µA เป็น mA (หารด้วย 1000)
                    current_ma = float(tube_current_in_ua_val) / 1000.0
                    tube_current_display_ma = int(round(current_ma)) # แสดงเป็น mA และปัดเป็นจำนวนเต็ม
                except (ValueError, TypeError):
                    tube_current_display_ma = str(tube_current_in_ua_val) # ถ้าแปลงไม่ได้ ให้แสดงค่าดิบ
            # --- สิ้นสุดส่วนที่ปรับปรุง ---
            
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

# ==============================================================================
# ==== FLASK ROUTES ====
# ==============================================================================

@app.route('/', methods=['GET', 'POST'])
def select_modality():
    """Handles modality selection and clears data if modality changes."""
    modalities = ['CT', 'DX'] # MG is a placeholder, no specific extraction logic
    if request.method == 'POST':
        selected_modality = request.form.get('modality')
        if selected_modality in modalities:
            current_session_modality = session.get('modality')
            if current_session_modality != selected_modality:
                REPORT_INDEX.clear()
                clear_folder(app.config['UPLOAD_FOLDER'], extensions_to_delete=['.dcm'])
                clear_folder(STATIC_FOLDER, extensions_to_delete=['.png']) # Clears all pngs, be cautious
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
    """Renders the file upload page for the selected modality."""
    modality = session.get('modality')
    if not modality:
        flash("Please select a modality first.", "warning")
        return redirect(url_for('select_modality'))
    return render_template('index.html', modality=modality)

@app.route('/process', methods=['POST'])
def process_files():
    """Processes uploaded DICOM files."""
    modality = session.get('modality')
    if not modality:
        flash("Modality not selected. Please select a modality first.", "error")
        return redirect(url_for('select_modality'))

    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        flash("No files were selected for upload.", "warning")
        return redirect(url_for('index'))

    errors, processed_count = [], 0
    for file_storage in uploaded_files: # Renamed 'file' to 'file_storage' to avoid conflict
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
                errors.append(f"File '{original_filename}' is {file_actual_modality}, but {modality} selected. Skipped.")
                if os.path.exists(temp_path): os.remove(temp_path)
                continue

            if modality == 'CT': main_info, _, _ = extract_ct_dose_info(ds)
            elif modality == 'DX': main_info = extract_dx_dose_info(ds)
            else:
                errors.append(f"Processing logic for modality '{modality}' not implemented for '{original_filename}'.")
                if os.path.exists(temp_path): os.remove(temp_path)
                continue

            report_id = f"{unique_suffix}_{os.path.splitext(original_filename)[0]}"
            report_data = {
                'id': report_id, 'filename': original_filename, 'save_name': save_name, 'modality': modality,
                'Patient ID': main_info.get('Patient ID', 'N/A'),
                'Study Date': main_info.get('Study Date', 'N/A'), # Already formatted
                'Raw Study Date': get_clean_value(ds, (0x0008,0x0020)) # Store raw from ds
            }
            
            if modality == 'CT':
                report_data.update({
                    'study_description': main_info.get('Study Description', 'N/A'),
                    'series_description': main_info.get('Series Description', 'N/A'),
                    'patient_age': main_info.get('Patient Age', 'N/A')
                })
            elif modality == 'DX':
                report_data.update({
                    'body_part': main_info.get('Body Part Examined', 'N/A'), # Already normalized
                    'view_position': main_info.get('View Position', 'N/A'),
                    'exposure': main_info.get('Exposure (mAs)', 'N/A'),
                    'kvp': main_info.get('kVp', 'N/A'),
                    TARGET_DAP_STORAGE_KEY_DX: main_info.get(TARGET_DAP_UNIT_LABEL_DX, 'N/A')
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
    elif not errors and any(fs.filename for fs in uploaded_files): # Files selected but none processed
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
    
    # --- ส่วนที่เพิ่มเข้ามาสำหรับการเรียงลำดับ ---
    sort_by = request.args.get('sort_by', 'Study Date') # ค่า Default คือเรียงตาม Study Date
    sort_order = request.args.get('sort_order', 'desc') # ค่า Default คือเรียงจากมากไปน้อย (ใหม่ไปเก่า)
    
    # ตรวจสอบว่า sort_order เป็นค่าที่ถูกต้อง
    if sort_order not in ['asc', 'desc']:
        sort_order = 'desc'
    # --- สิ้นสุดส่วนที่เพิ่มเข้ามา ---

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
            
            if (matches_base or matches_modality_specific) and r['id'] not in matching_report_ids:
                temp_filtered_reports.append(r)
                matching_report_ids.add(r['id'])
        final_filtered_reports = temp_filtered_reports
    else:
        final_filtered_reports = relevant_reports
        
    # --- ส่วนที่เพิ่มเข้ามาสำหรับการเรียงลำดับข้อมูล ---
    if final_filtered_reports: # ตรวจสอบว่ามีข้อมูลให้เรียงหรือไม่
        # กำหนด key สำหรับการเรียงลำดับ
        # คุณอาจจะต้องปรับ key ให้ตรงกับชื่อ key ใน REPORT_INDEX ของคุณ
        # และจัดการกับการแปลงค่าถ้าจำเป็น (เช่น วันที่, ตัวเลข)
        
        # ตัวอย่างการจัดการกับการเรียงตามวันที่ (Study Date)
        # เราจะเรียงตาม 'Raw Study Date' (YYYYMMDD) เพื่อให้เรียงถูกต้อง
        # หรือถ้า 'Study Date' ของคุณเป็น YYYY-MM-DD ก็ใช้ได้เลย
        
        is_reverse = (sort_order == 'desc')

        # สร้าง lambda function สำหรับการดึงค่า key โดยจัดการกับ None หรือค่าที่ไม่มีอยู่
        def sort_key_func(report):
            val = report.get(sort_by)
            if val is None: # จัดการกับ None ให้น้อยที่สุดเมื่อเรียงจากน้อยไปมาก
                return float('-inf') if sort_order == 'asc' else float('inf')
            
            # แปลงค่า DAP เป็นตัวเลขสำหรับการเรียงที่ถูกต้อง
            if sort_by == TARGET_DAP_STORAGE_KEY_DX: #
                try:
                    return float(val)
                except (ValueError, TypeError): # ถ้าแปลงไม่ได้ ให้ถือว่าน้อยที่สุด/มากที่สุด
                    return float('-inf') if sort_order == 'asc' else float('inf')
            
            # สำหรับ Study Date ควรใช้ Raw Study Date (YYYYMMDD) เพื่อการเรียงที่ถูกต้อง
            if sort_by == 'Study Date' and 'Raw Study Date' in report:
                 raw_date = report.get('Raw Study Date')
                 if raw_date and str(raw_date).isdigit() and len(str(raw_date)) == 8:
                     return raw_date
                 # Fallback ถ้า Raw Study Date ไม่มีรูปแบบที่คาดหวัง
                 return val if val is not None else (float('-inf') if sort_order == 'asc' else float('inf'))

            # สำหรับค่าอื่นๆ ที่เป็นสตริง
            if isinstance(val, str):
                return val.lower() # เรียงสตริงแบบไม่สน case
            return val

        try:
            final_filtered_reports.sort(key=sort_key_func, reverse=is_reverse)
        except TypeError as e:
            flash(f"Could not sort by '{sort_by}'. Ensure data types are consistent or add specific handling. Error: {e}", "warning")
            # อาจจะ fallback ไปเรียงแบบ default หรือไม่เรียงเลย
            # ในที่นี้จะปล่อยให้ข้อมูลไม่ถูกเรียงถ้าเกิด TypeError
            pass
    # --- สิ้นสุดส่วนที่เพิ่มเข้ามา ---
            
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
    body_parts_dx = []
    if modality_filter == 'DX':
        body_parts_dx = sorted(list(set(
            r['body_part'] for r in relevant_reports 
            if r.get('body_part') and r.get('body_part','N/A') != 'N/A'
        )))

    return render_template(
        'report_list.html', reports=reports_on_page, page=page, pages=pages,
        search=search_term, total=total, current_modality=modality_filter,
        study_descriptions=study_descriptions_ct,
        body_parts_examined=body_parts_dx, 
        TARGET_DAP_STORAGE_KEY_DX=TARGET_DAP_STORAGE_KEY_DX, #
        PER_PAGE=PER_PAGE,
        # --- ส่งตัวแปร sort ไปยัง template ---
        current_sort_by=sort_by,
        current_sort_order=sort_order
        # --- สิ้นสุดการส่งตัวแปร ---
    )

@app.route('/delete_file/<report_id>', methods=['POST'])
def delete_file(report_id):
    """Deletes a report file and its entry from the index."""
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
        
    # Maintain pagination and search context on redirect
    page = request.form.get('page', 1, type=int)
    search = request.form.get('search', '')
    return redirect(url_for('list_reports', page=page, search=search))

# In app.py

# In app.py

@app.route('/report/<report_id>')
def view_report(report_id):
    report_meta = next((r for r in REPORT_INDEX if r['id'] == report_id), None)
    if not report_meta:
        flash(f"Report with ID '{report_id}' not found.", 'error')
        return redirect(url_for('list_reports'))
    
    errors_list = [] # Initialize errors_list at the top
    
    template_context = {
        'modality': report_meta['modality'],
        'filename': report_meta['filename'],
        'report_id': report_id,
        'info': None, 
        'dx_info': None, 
        'data': None, 
        'total_dlp': None,
        'errors': errors_list # Pass the list to the template; it will be updated
    }
    
    try:
        dicom_file_path = os.path.join(app.config['UPLOAD_FOLDER'], report_meta['save_name'])
        if not os.path.exists(dicom_file_path):
            msg = f"DICOM file '{report_meta['filename']}' (saved as '{report_meta['save_name']}') not found on disk."
            errors_list.append(msg)
            # Populate with metadata from REPORT_INDEX for partial view if file is missing
            if report_meta['modality'] == 'CT': template_context['info'] = report_meta
            elif report_meta['modality'] == 'DX': template_context['dx_info'] = report_meta
            # No flash here yet, will flash all accumulated errors at the end
        else:
            # File exists, proceed with reading
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
                msg = f"Unsupported modality '{report_meta['modality']}' for detailed view of '{report_meta['filename']}'."
                errors_list.append(msg)
    
    except pydicom.errors.InvalidDicomError:
        msg = f"File '{report_meta['filename']}' is not a valid DICOM format or is corrupted."
        errors_list.append(msg)
    except Exception as e:
        msg = f"An unexpected error occurred while reading report details for '{report_meta['filename']}': {type(e).__name__} - {str(e)}"
        errors_list.append(msg)
        print(f"Detailed error for {report_meta['filename']}: {type(e).__name__} - {e}") # Log full error

    # Now, flash all accumulated errors (if any) ONCE
    if errors_list:
        # You can flash each error individually or a combined message
        # For individual messages:
        # for error_msg in errors_list:
        #     flash(error_msg, "error")
        # For a combined message:
        combined_error_message = "; ".join(errors_list)
        flash(f"Issues loading report: {combined_error_message}", "error")
    
    template_context['errors'] = errors_list # Ensure the template context has the final list

    # If detailed info/dx_info was not populated due to an error,
    # try to populate with basic info from report_meta for partial display.
    if template_context['modality'] == 'CT' and not template_context['info'] and report_meta:
        template_context['info'] = report_meta 
    if template_context['modality'] == 'DX' and not template_context['dx_info'] and report_meta:
        template_context['dx_info'] = report_meta
        
    return render_template('results.html', **template_context)

# --- Comparison and Export Routes ---
def _generate_comparison_plot(x_data, y_data, x_label, y_label, title_prefix, group_name):
    """Helper function to generate and save a bar plot."""
    plt.figure(figsize=(max(8, len(x_data) * 0.5 + 2), 6.5))
    plt.bar(x_data, y_data, color='#5eaaa8', width=0.6) # Consistent color
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
    """Helper to create and send an Excel file from a DataFrame."""
    try:
        output_stream = BytesIO()
        df.to_excel(output_stream, index=False, sheet_name=sheet_name)
        output_stream.seek(0)
        
        safe_filename_suffix = re.sub(r'[^a-zA-Z0-9_.-]', '_', sheet_name) # Use sheet_name for suffix if relevant
        download_filename = f'{download_base_filename}_{safe_filename_suffix}.xlsx'
        
        return send_file(
            output_stream, as_attachment=True, download_name=download_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        flash(f"Error generating Excel file: {e}", "error")
        return None


@app.route('/compare_dlp')
def compare_dlp():
    """Compares Total DLP for CT reports, grouped by Study Description."""
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
            _, _, total_dlp_val = extract_ct_dose_info(ds) # Use extraction function

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
    """Exports filtered CT DLP data to Excel."""
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


@app.route('/compare_dap')
def compare_dap():
    """Compares DAP for DX reports, grouped by Body Part, aggregated by Patient."""
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
                'Body Part Examined': r_meta.get('body_part','').strip().upper(), # Normalized
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
    """Exports filtered DX DAP data to Excel."""
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

@app.route('/search_patient', methods=['GET', 'POST'])
def search_patient():
    """Searches for studies by Patient ID for the current modality."""
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
                    entry['total_dlp'] = 'N/A' # Default
                    try: # Re-read for TotalDLP from comments
                        dicom_path = os.path.join(app.config['UPLOAD_FOLDER'], r_meta['save_name'])
                        if os.path.exists(dicom_path):
                            ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
                            _, _, total_dlp_val = extract_ct_dose_info(ds)
                            if total_dlp_val is not None: entry['total_dlp'] = total_dlp_val
                    except Exception as e: print(f"Error getting TotalDLP for search: {e}")
                elif current_modality == 'DX':
                    entry['body_part'] = r_meta.get('body_part', 'N/A')
                    entry['dap_value_label'] = TARGET_DAP_UNIT_LABEL_DX # For template
                    entry['dap_value'] = r_meta.get(TARGET_DAP_STORAGE_KEY_DX, 'N/A')
                results.append(entry)
            if not results: flash(f"No {current_modality} reports for Patient ID '{patient_id_search}'.", "info")
        elif request.method == 'POST': flash("Enter Patient ID to search.", "warning")
            
    return render_template('search_patient.html', results=results, 
                           patient_id=patient_id_search, current_modality=current_modality)

@app.route('/compare_patient/<patient_id>')
def compare_patient(patient_id):
    """Compares all studies for a given patient for the current modality."""
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
            try: # Re-read for TotalDLP
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
            if not str(dap_str).lower().startswith('n/a'):
                try: numeric_val = float(dap_str)
                except ValueError: pass
        entry['value'] = numeric_val
        studies_data.append(entry) # Add all studies for table display

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
    # For production, set debug=False and use a production-ready WSGI server (e.g., Gunicorn, Waitress)
    app.run(debug=True, host='0.0.0.0', port=port)
