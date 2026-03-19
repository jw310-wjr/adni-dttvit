import os
import subprocess
import logging
import ants
import pandas as pd

# -------------------------------
# Configuration and Directory Setup
# -------------------------------

BASE_DIR = "/scratch/jw310/adni-dttvit/data"
CSV_FILE = f"{BASE_DIR}/merged431.csv"


skull_stripped_dir = f"{BASE_DIR}/stripped"
affine_reg_dir     = f"{BASE_DIR}/affine"
syn_reg_dir        = f"{BASE_DIR}/syn"
jacobian_dir       = f"{BASE_DIR}/jacobian"

SYNTHSTRIP_PY    = "/scratch/jw310/tools/synthstrip/synthstrip.py"
SYNTHSTRIP_MODEL = "/scratch/jw310/tools/synthstrip/synthstrip.1.pt"
MNI_TEMPLATE = "/scratch/jw310/tools/templates/mni152_t1_1mm.nii.gz"


GPU_ENABLED = True
PROCESS_TIMEOUT = 600
REG_TIMEOUT = 3600   # 1 hour

# -------------------------------
# Logging Configuration
# -------------------------------
from datetime import datetime

RUN_NAME = f"mni_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR  = f"/scratch/jw310/adni-dttvit/runs/active/{RUN_NAME}"
os.makedirs(RUN_DIR, exist_ok=False)   

LOG_FILE = os.path.join(RUN_DIR, "MNI_preprocessing.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
    force=True
)

# -------------------------------
# Initialization and Environment Validation
# -------------------------------

def validate_environment():
    required_paths = [
    ('CSV file', CSV_FILE),
    ('Skull-stripped output directory', skull_stripped_dir),
    ('SynthStrip script', SYNTHSTRIP_PY),
    ('SynthStrip model', SYNTHSTRIP_MODEL),
    ('MNI template', MNI_TEMPLATE),
]


    for label, path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
        if 'directory' in label and not os.path.isdir(path):
            raise NotADirectoryError(f"{label} is not a directory: {path}")

    global GPU_ENABLED
    if GPU_ENABLED:
        try:
            subprocess.run(["nvidia-smi"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logging.info("GPU access verified")
        except Exception:
            logging.warning("GPU not available. Falling back to CPU mode.")
            GPU_ENABLED = False

    try:
        fixed = ants.image_read(MNI_TEMPLATE)
        logging.info("Fixed MNI template loaded successfully.")
        return fixed
    except Exception as e:
        logging.error(f"Failed to load fixed template: {e}")
        return None

# -------------------------------
# Core Processing Functions
# -------------------------------

def skull_strip(input_path, base_subject):
    output_path = os.path.join(skull_stripped_dir, f"{base_subject}_brain.nii.gz")
    mask_path   = os.path.join(skull_stripped_dir, f"{base_subject}_mask.nii.gz")

    if (os.path.exists(output_path) and os.path.getsize(output_path) >= 1024 and
        os.path.exists(mask_path)   and os.path.getsize(mask_path)   >= 1024):
        logging.info(f"Skull stripped outputs already exist for {base_subject}. Skipping.")
        return output_path

    logging.info(f"Skull stripping: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")

    cmd = [
        "python", SYNTHSTRIP_PY,
        "-i", input_path,
        "-o", output_path,
        "-m", mask_path,
        "--model", SYNTHSTRIP_MODEL,
    ]
    if GPU_ENABLED:
        cmd += ["--gpu"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=PROCESS_TIMEOUT)

        # sanity checks
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Output not created: {output_path}")
        if os.path.getsize(output_path) < 1024:
            raise ValueError(f"Suspiciously small output: {output_path}")
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) < 1024:
            raise ValueError(f"Suspiciously small/missing mask: {mask_path}")

        logging.info(f"Skull stripping completed for {base_subject}")
        return output_path

    except subprocess.TimeoutExpired:
        logging.error(f"Timeout processing {os.path.basename(input_path)}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Skull stripping failed for {os.path.basename(input_path)}:\nSTDERR:\n{e.stderr}")
        raise



def process_image(row, fixed_template):
    subject_id = str(row["Subject"]).strip()

    raw_image_path = str(row["nifti_path"]).strip()

    if not raw_image_path.startswith("/"):
        raw_image_path = os.path.join("/scratch/jw310/adni-dttvit", raw_image_path)

    if not os.path.exists(raw_image_path):
        logging.error(f"Raw image not found for {subject_id}: {raw_image_path}")
        return False

    logging.info(f"Processing Subject: {subject_id}")

    base_subject = subject_id + "_m12"
    jac_path = os.path.join(jacobian_dir, f"jacobian_{base_subject}.nii.gz")

    if os.path.exists(jac_path) and os.path.getsize(jac_path) >= 1024:
        logging.info(f"Jacobian already exists for {subject_id} at {jac_path}. Skipping.")
        return True

    if fixed_template is None:
        logging.error(f"Fixed template missing. Skipping registration for {subject_id}.")
        return False


    # --------------------
    # (1) N4 bias correction
    # --------------------
    try:
        n4_dir = os.path.join(BASE_DIR, "n4")
        os.makedirs(n4_dir, exist_ok=True)
        corrected_path = os.path.join(n4_dir, f"{base_subject}_N4.nii.gz")

        if os.path.exists(corrected_path) and os.path.getsize(corrected_path) >= 1024:
            logging.info(f"N4 already exists for {subject_id}. Using {corrected_path}")
        else:
            raw_img = ants.image_read(raw_image_path)
            logging.info(f"Performing N4 bias correction for {subject_id}")
            corrected_img = ants.n4_bias_field_correction(raw_img)
            ants.image_write(corrected_img, corrected_path)
            logging.info(f"N4 bias correction saved to {corrected_path}")

        image_for_strip = corrected_path
    except Exception as e:
        logging.error(f"N4 bias correction failed for {subject_id}: {e}")
        return False

    # --------------------
    # (2) Skull stripping
    # --------------------
    try:
        stripped_path = skull_strip(image_for_strip, base_subject)
    except Exception as e:
        logging.error(f"Skull stripping failed for {subject_id}: {e}")
        return

    try:
        moving_image = ants.image_read(stripped_path)
    except Exception as e:
        logging.error(f"Failed to read skull-stripped image for {subject_id}: {e}")
        return False

    corrected = moving_image

    # --------------------
    # (3) Affine registration
    # --------------------
    try:
        logging.info(f"Performing Affine registration for {subject_id}")
        affine_result = ants.registration(
            fixed=fixed_template,
            moving=corrected,
            type_of_transform='Affine',
            verbose=False
        )
        affine_path = os.path.join(affine_reg_dir, f"affine_registered_{base_subject}.nii.gz")
        ants.image_write(affine_result['warpedmovout'], affine_path)
        logging.info(f"Affine registration saved to {affine_path}")
    except Exception as e:
        logging.error(f"Affine registration failed for {subject_id}: {e}")
        return False

    # --------------------
    # (4) SyN registration
    # --------------------
    try:
        logging.info(f"Performing SyN registration for {subject_id}")

        # initial_transform: use affine final transform (affine.mat)
        aff_fwd = affine_result.get('fwdtransforms', [])
        init_tx = aff_fwd[-1] if isinstance(aff_fwd, (list, tuple)) and len(aff_fwd) > 0 else None

        syn_kwargs = dict(
            fixed=fixed_template,
            moving=corrected,
            type_of_transform='SyN',
            verbose=False
        )
        if init_tx is not None:
            syn_kwargs["initial_transform"] = init_tx  

        syn_result = ants.registration(**syn_kwargs)

        syn_path = os.path.join(syn_reg_dir, f"syn_registered_{base_subject}.nii.gz")
        ants.image_write(syn_result['warpedmovout'], syn_path)
        logging.info(f"SyN registration saved to {syn_path}")
    except Exception as e:
        logging.error(f"SyN registration failed for {subject_id}: {e}")
        return False

    # --------------------
    # (5) Jacobian
    # --------------------
    try:
        fwd = syn_result.get('fwdtransforms', [])
        if not fwd:
            raise ValueError("syn_result has no fwdtransforms")

        # warp (.nii/.nii.gz), avoid affine.mat
        warp = next((t for t in fwd if str(t).endswith(".nii") or str(t).endswith(".nii.gz")), None)
        if warp is None:
            raise ValueError(f"No warp found in fwdtransforms: {fwd}")

        logging.info(f"Computing Jacobian for {subject_id}")
        jacobian = ants.create_jacobian_determinant_image(
            domain_image=fixed_template,
            tx=warp,
            do_log=True
        )
        ants.image_write(jacobian, jac_path)
        logging.info(f"Jacobian saved to {jac_path}")
    except Exception as e:
        logging.error(f"Jacobian calculation failed for {subject_id}: {e}")
        return False

    logging.info(f"Completed processing for {subject_id}")
    return True



# -------------------------------
# Main Runner
# -------------------------------

def main():
    fixed_template = validate_environment()
    if fixed_template is None:
        logging.error("Cannot proceed without a valid fixed template.")
        return

    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip()
        n = len(df)
        logging.info(f"CSV file read successfully. {n} records found.")
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return

    success, fail = 0, 0

    for idx, row in df.iterrows():

        subject = row.get("Subject", "UNKNOWN")
        logging.info(f"Processing record {idx+1}/{n}: Subject {subject}")

        ok = process_image(row, fixed_template)

        if ok:
            success += 1
        else:
            fail += 1

    logging.info(f"All processing complete. success={success}, fail={fail}")




if __name__ == "__main__":
    main()

