import base64
import io
import time
from typing import Any, Dict, List

from flask import Flask, Response, g, redirect, render_template, request, url_for
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from model_client import (
    ModelServiceError,
    fetch_model_service_health,
    get_model_api_url,
    predict_with_model_service,
)

APP_NAME = "Cancer Hospital Management App"
TODAY = "2026-04-08"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

METRICS_EXCLUDED_PATH_PREFIXES = ("/metrics", "/healthz", "/static/")

HTTP_REQUESTS_TOTAL = Counter(
    "cancer_hospital_http_requests_total",
    "Total HTTP requests handled by the app.",
    ["method", "route", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "cancer_hospital_http_request_duration_seconds",
    "Request latency by route.",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
INFERENCE_TOTAL = Counter(
    "cancer_hospital_inference_total",
    "Number of pathology inference operations.",
    ["stage", "status"],
)
INFERENCE_DURATION_SECONDS = Histogram(
    "cancer_hospital_inference_duration_seconds",
    "Latency of prediction and Grad-CAM stages.",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
MODEL_ARTIFACT_READY = Gauge(
    "cancer_hospital_model_artifact_ready",
    "Whether required model artifacts are available in the host-side model service.",
    ["artifact"],
)
MODEL_SERVICE_AVAILABLE = Gauge(
    "cancer_hospital_model_service_available",
    "Whether the host-side model service can be reached.",
)
GPU_AVAILABLE = Gauge(
    "cancer_hospital_gpu_available",
    "Whether a GPU backend is available to the host-side model service.",
    ["backend"],
)
GPU_DEVICE_COUNT = Gauge(
    "cancer_hospital_gpu_device_count",
    "Number of CUDA devices visible to the host-side model service.",
)
GPU_MEMORY_ALLOCATED_BYTES = Gauge(
    "cancer_hospital_gpu_memory_allocated_bytes",
    "CUDA memory allocated by the host-side model service.",
)
GPU_MEMORY_RESERVED_BYTES = Gauge(
    "cancer_hospital_gpu_memory_reserved_bytes",
    "CUDA memory reserved by the host-side model service.",
)


DOCTORS = [
    {
        "id": "DR-201",
        "name": "Dr. Aarya Menon",
        "role": "Consultant Oncopathologist",
        "specialty": "Lung and GI pathology",
        "hospital": "Meridian Cancer Centre",
        "email": "aarya.menon@meridiancare.org",
        "phone": "+91 22 4100 1001",
        "patients_today": 18,
        "pending_reviews": 5,
        "critical_cases": 2,
        "assigned_patient_ids": ["PT-1042", "PT-1048"],
    },
    {
        "id": "DR-202",
        "name": "Dr. Kunal Bhasin",
        "role": "Thoracic Oncologist",
        "specialty": "Lung cancer management",
        "hospital": "Meridian Cancer Centre",
        "email": "kunal.bhasin@meridiancare.org",
        "phone": "+91 22 4100 1002",
        "patients_today": 14,
        "pending_reviews": 3,
        "critical_cases": 1,
        "assigned_patient_ids": ["PT-1042", "PT-1051"],
    },
    {
        "id": "DR-203",
        "name": "Dr. Meera Sethi",
        "role": "GI Medical Oncologist",
        "specialty": "Colorectal oncology",
        "hospital": "Meridian Cancer Centre",
        "email": "meera.sethi@meridiancare.org",
        "phone": "+91 22 4100 1003",
        "patients_today": 16,
        "pending_reviews": 4,
        "critical_cases": 1,
        "assigned_patient_ids": ["PT-1048", "PT-1060"],
    },
    {
        "id": "DR-204",
        "name": "Dr. Nisha Rao",
        "role": "Radiation Oncologist",
        "specialty": "Planning and follow-up",
        "hospital": "Meridian Cancer Centre",
        "email": "nisha.rao@meridiancare.org",
        "phone": "+91 22 4100 1004",
        "patients_today": 11,
        "pending_reviews": 2,
        "critical_cases": 0,
        "assigned_patient_ids": ["PT-1051", "PT-1060"],
    },
]

DOCTORS.extend(
    [
        {
            "id": "DR-205",
            "name": "Dr. Vikram Sen",
            "role": "Surgical Oncologist",
            "specialty": "Colorectal surgery",
            "hospital": "Meridian Cancer Centre",
            "email": "vikram.sen@meridiancare.org",
            "phone": "+91 22 4100 1005",
            "patients_today": 12,
            "pending_reviews": 3,
            "critical_cases": 1,
            "assigned_patient_ids": ["PT-1064", "PT-1072", "PT-1092"],
        },
        {
            "id": "DR-206",
            "name": "Dr. Ritu Kapoor",
            "role": "Pulmonologist",
            "specialty": "Interventional pulmonary medicine",
            "hospital": "Meridian Cancer Centre",
            "email": "ritu.kapoor@meridiancare.org",
            "phone": "+91 22 4100 1006",
            "patients_today": 13,
            "pending_reviews": 4,
            "critical_cases": 2,
            "assigned_patient_ids": ["PT-1068", "PT-1088"],
        },
        {
            "id": "DR-207",
            "name": "Dr. Farah Khan",
            "role": "Supportive Care Oncologist",
            "specialty": "Nutrition and symptom support",
            "hospital": "Meridian Cancer Centre",
            "email": "farah.khan@meridiancare.org",
            "phone": "+91 22 4100 1007",
            "patients_today": 10,
            "pending_reviews": 2,
            "critical_cases": 1,
            "assigned_patient_ids": ["PT-1042", "PT-1064", "PT-1068", "PT-1072"],
        },
        {
            "id": "DR-208",
            "name": "Dr. Dev Malhotra",
            "role": "Nuclear Medicine Specialist",
            "specialty": "Staging and treatment imaging",
            "hospital": "Meridian Cancer Centre",
            "email": "dev.malhotra@meridiancare.org",
            "phone": "+91 22 4100 1008",
            "patients_today": 9,
            "pending_reviews": 2,
            "critical_cases": 1,
            "assigned_patient_ids": ["PT-1042", "PT-1060", "PT-1076", "PT-1088"],
        },
        {
            "id": "DR-209",
            "name": "Dr. Ishita Ghosh",
            "role": "Medical Oncologist",
            "specialty": "Systemic therapy planning",
            "hospital": "Meridian Cancer Centre",
            "email": "ishita.ghosh@meridiancare.org",
            "phone": "+91 22 4100 1009",
            "patients_today": 15,
            "pending_reviews": 4,
            "critical_cases": 2,
            "assigned_patient_ids": ["PT-1060", "PT-1076", "PT-1084"],
        },
        {
            "id": "DR-210",
            "name": "Dr. Naveen Kulkarni",
            "role": "Senior Oncopathologist",
            "specialty": "Slide review and AI correlation",
            "hospital": "Meridian Cancer Centre",
            "email": "naveen.kulkarni@meridiancare.org",
            "phone": "+91 22 4100 1010",
            "patients_today": 17,
            "pending_reviews": 5,
            "critical_cases": 2,
            "assigned_patient_ids": ["PT-1048", "PT-1068", "PT-1080"],
        },
    ]
)

PATIENTS = [
    {
        "id": "PT-1042",
        "name": "Rohan Sharma",
        "age": 54,
        "sex": "Male",
        "status": "Priority review",
        "clinic": "Thoracic Oncology",
        "specimen": "Right lung biopsy",
        "scheduled_time": "09:20",
        "risk_level": "High",
        "diagnosis_stage": "Workup pending",
        "last_prediction": "lung_aca",
        "last_confidence": 97.8,
        "mrn": "MCC-88214",
        "doctor_id": "DR-201",
        "ward": "Thoracic Day Care",
        "case_summary": "Progressive cough, weight loss, and a suspicious upper lobe lesion on CT.",
        "allergies": ["Penicillin"],
        "comorbidities": ["Type 2 diabetes", "Hypertension"],
        "medications": ["Metformin 500 mg", "Amlodipine 5 mg"],
        "history": [
            {"date": "2026-04-01", "event": "CT chest reviewed", "detail": "Spiculated lesion in right upper lobe."},
            {"date": "2026-04-04", "event": "Biopsy accessioned", "detail": "Histopathology slides prepared for review."},
            {"date": "2026-04-07", "event": "Tumor board prep", "detail": "Awaiting pathology correlation."},
        ],
    },
    {
        "id": "PT-1048",
        "name": "Meera Iyer",
        "age": 47,
        "sex": "Female",
        "status": "Routine follow-up",
        "clinic": "GI Oncology",
        "specimen": "Colon resection block",
        "scheduled_time": "10:05",
        "risk_level": "Moderate",
        "diagnosis_stage": "Post-op review",
        "last_prediction": "colon_aca",
        "last_confidence": 95.4,
        "mrn": "MCC-88277",
        "doctor_id": "DR-203",
        "ward": "GI Oncology Floor",
        "case_summary": "Post-operative pathology review after left hemicolectomy for known lesion.",
        "allergies": ["None known"],
        "comorbidities": ["Iron deficiency anemia"],
        "medications": ["Ferrous sulfate", "Pantoprazole"],
        "history": [
            {"date": "2026-03-24", "event": "Colonoscopy biopsy", "detail": "Moderately differentiated adenocarcinoma suspected."},
            {"date": "2026-03-30", "event": "Surgery completed", "detail": "Specimen submitted for final pathology."},
            {"date": "2026-04-06", "event": "CEA trend reviewed", "detail": "CEA mildly elevated, stable post-op."},
        ],
    },
    {
        "id": "PT-1051",
        "name": "Arjun Patel",
        "age": 62,
        "sex": "Male",
        "status": "Benign correlation",
        "clinic": "Pulmonary Clinic",
        "specimen": "Left lung wedge biopsy",
        "scheduled_time": "11:10",
        "risk_level": "Low",
        "diagnosis_stage": "Correlation review",
        "last_prediction": "lung_n",
        "last_confidence": 93.2,
        "mrn": "MCC-88301",
        "doctor_id": "DR-202",
        "ward": "Pulmonary Observation",
        "case_summary": "Benign inflammatory changes suspected; pathology requested to exclude malignancy.",
        "allergies": ["Sulfa drugs"],
        "comorbidities": ["COPD"],
        "medications": ["Tiotropium inhaler", "Albuterol PRN"],
        "history": [
            {"date": "2026-03-28", "event": "PET reviewed", "detail": "Low avidity lesion, indeterminate."},
            {"date": "2026-04-03", "event": "Biopsy taken", "detail": "Small wedge biopsy with preserved tissue architecture."},
            {"date": "2026-04-07", "event": "Pulmonary review", "detail": "Favor benign reactive process clinically."},
        ],
    },
    {
        "id": "PT-1060",
        "name": "Sana Qureshi",
        "age": 39,
        "sex": "Female",
        "status": "Chemotherapy planning",
        "clinic": "Day Oncology",
        "specimen": "Colon biopsy follow-up",
        "scheduled_time": "12:25",
        "risk_level": "High",
        "diagnosis_stage": "Treatment planning",
        "last_prediction": "colon_aca",
        "last_confidence": 98.1,
        "mrn": "MCC-88344",
        "doctor_id": "DR-203",
        "ward": "Medical Oncology Bay",
        "case_summary": "Known colon adenocarcinoma under staging review for treatment pathway selection.",
        "allergies": ["Latex"],
        "comorbidities": ["Hypothyroidism"],
        "medications": ["Levothyroxine 50 mcg", "Ondansetron PRN"],
        "history": [
            {"date": "2026-03-27", "event": "Biopsy confirmed", "detail": "Adenocarcinoma confirmed on pathology."},
            {"date": "2026-04-02", "event": "Staging CT done", "detail": "No distant metastasis reported."},
            {"date": "2026-04-07", "event": "Chemo counseling", "detail": "Patient counseled on adjuvant options."},
        ],
    },
]

PATIENTS.extend(
    [
        {
            "id": "PT-1064",
            "name": "Nidhi Verma",
            "age": 58,
            "sex": "Female",
            "status": "Treatment planning",
            "clinic": "GI Surgical Oncology",
            "specimen": "Colon biopsy panel",
            "scheduled_time": "13:00",
            "risk_level": "High",
            "diagnosis_stage": "Pre-adjuvant review",
            "last_prediction": "colon_aca",
            "last_confidence": 97.1,
            "mrn": "MCC-88372",
            "doctor_id": "DR-205",
            "ward": "GI Oncology Floor",
            "case_summary": "Ascending colon lesion under surgical review with anemia and altered bowel habits.",
            "allergies": ["None known"],
            "comorbidities": ["Iron deficiency anemia"],
            "medications": ["Ferrous sulfate", "Pantoprazole"],
            "history": [
                {"date": "2026-04-02", "event": "Biopsy slides reviewed", "detail": "Gland-forming malignant pattern under correlation."},
                {"date": "2026-04-05", "event": "Surgical consult", "detail": "Operability reviewed with colorectal team."},
                {"date": "2026-04-07", "event": "Dietician referral", "detail": "Protein support discussed before treatment start."},
            ],
        },
        {
            "id": "PT-1068",
            "name": "Kabir Singh",
            "age": 65,
            "sex": "Male",
            "status": "Urgent thoracic review",
            "clinic": "Thoracic Oncology",
            "specimen": "Bronchial biopsy",
            "scheduled_time": "13:20",
            "risk_level": "High",
            "diagnosis_stage": "Urgent staging",
            "last_prediction": "lung_scc",
            "last_confidence": 96.6,
            "mrn": "MCC-88401",
            "doctor_id": "DR-206",
            "ward": "Thoracic Day Care",
            "case_summary": "Central airway lesion with persistent cough and tobacco exposure history.",
            "allergies": ["NSAIDs"],
            "comorbidities": ["Chronic bronchitis"],
            "medications": ["Albuterol PRN", "Budesonide inhaler"],
            "history": [
                {"date": "2026-04-01", "event": "Bronchoscopy done", "detail": "Endobronchial lesion sampled for pathology."},
                {"date": "2026-04-04", "event": "Respiratory review", "detail": "Cough burden worsening overnight."},
                {"date": "2026-04-07", "event": "Oncology triage", "detail": "Marked for same-week MDT discussion."},
            ],
        },
        {
            "id": "PT-1072",
            "name": "Pooja Nair",
            "age": 44,
            "sex": "Female",
            "status": "Surveillance review",
            "clinic": "GI Follow-up",
            "specimen": "Colon mucosal biopsy",
            "scheduled_time": "13:45",
            "risk_level": "Low",
            "diagnosis_stage": "Benign correlation",
            "last_prediction": "colon_n",
            "last_confidence": 92.8,
            "mrn": "MCC-88419",
            "doctor_id": "DR-205",
            "ward": "GI Day Observation",
            "case_summary": "Inflammatory bowel symptoms under surveillance after benign biopsy pattern.",
            "allergies": ["Shellfish"],
            "comorbidities": ["IBS"],
            "medications": ["Dicyclomine PRN", "Probiotic sachet"],
            "history": [
                {"date": "2026-03-29", "event": "Colonoscopy repeated", "detail": "Patchy mucosal inflammation noted."},
                {"date": "2026-04-03", "event": "Pathology requested", "detail": "Asked to exclude dysplasia or malignancy."},
                {"date": "2026-04-07", "event": "Supportive care review", "detail": "Diet modification advised."},
            ],
        },
        {
            "id": "PT-1076",
            "name": "Imran Ali",
            "age": 59,
            "sex": "Male",
            "status": "Staging workup",
            "clinic": "Medical Oncology",
            "specimen": "Peripheral lung core biopsy",
            "scheduled_time": "14:10",
            "risk_level": "High",
            "diagnosis_stage": "Molecular workup pending",
            "last_prediction": "lung_aca",
            "last_confidence": 98.0,
            "mrn": "MCC-88444",
            "doctor_id": "DR-209",
            "ward": "Medical Oncology Bay",
            "case_summary": "Peripheral lung lesion suspicious for adenocarcinoma awaiting molecular staging.",
            "allergies": ["None known"],
            "comorbidities": ["Hypertension"],
            "medications": ["Telmisartan 40 mg"],
            "history": [
                {"date": "2026-04-02", "event": "Core biopsy obtained", "detail": "Adequate tissue sent for pathology and markers."},
                {"date": "2026-04-06", "event": "CT reviewed", "detail": "No pleural effusion seen on current scan."},
                {"date": "2026-04-07", "event": "Medical oncology visit", "detail": "Discussed systemic therapy options pending final sign-out."},
            ],
        },
        {
            "id": "PT-1080",
            "name": "Leena Joseph",
            "age": 51,
            "sex": "Female",
            "status": "Case correlation",
            "clinic": "Pathology Correlation",
            "specimen": "Colon resection slide",
            "scheduled_time": "14:35",
            "risk_level": "Moderate",
            "diagnosis_stage": "Final review pending",
            "last_prediction": "colon_aca",
            "last_confidence": 95.9,
            "mrn": "MCC-88462",
            "doctor_id": "DR-210",
            "ward": "GI Oncology Floor",
            "case_summary": "Post-resection slide set awaiting final consultant sign-out and margin correlation.",
            "allergies": ["Latex"],
            "comorbidities": ["Hypothyroidism"],
            "medications": ["Levothyroxine 75 mcg"],
            "history": [
                {"date": "2026-03-31", "event": "Surgery complete", "detail": "Specimen transferred for final pathology staging."},
                {"date": "2026-04-05", "event": "Margin review", "detail": "Close margin flagged for re-check."},
                {"date": "2026-04-07", "event": "AI correlation requested", "detail": "Consultant requested fused-model support."},
            ],
        },
        {
            "id": "PT-1084",
            "name": "Manav Deshpande",
            "age": 61,
            "sex": "Male",
            "status": "Pulmonary follow-up",
            "clinic": "Pulmonary Clinic",
            "specimen": "Lung wedge section",
            "scheduled_time": "15:00",
            "risk_level": "Low",
            "diagnosis_stage": "Benign follow-up",
            "last_prediction": "lung_n",
            "last_confidence": 94.4,
            "mrn": "MCC-88488",
            "doctor_id": "DR-209",
            "ward": "Pulmonary Observation",
            "case_summary": "Benign lung pattern under clinical follow-up after indeterminate imaging.",
            "allergies": ["Sulfa drugs"],
            "comorbidities": ["COPD", "GERD"],
            "medications": ["Tiotropium inhaler", "Pantoprazole"],
            "history": [
                {"date": "2026-04-01", "event": "Imaging review", "detail": "Lesion low avidity and likely inflammatory."},
                {"date": "2026-04-04", "event": "Biopsy review", "detail": "No overt malignant architecture appreciated."},
                {"date": "2026-04-07", "event": "Pulmonary call", "detail": "Stable symptoms and no new alarm signs."},
            ],
        },
        {
            "id": "PT-1088",
            "name": "Tara Bose",
            "age": 49,
            "sex": "Female",
            "status": "Rapid AI review",
            "clinic": "Thoracic Oncology",
            "specimen": "Lung biopsy slide",
            "scheduled_time": "15:20",
            "risk_level": "High",
            "diagnosis_stage": "Rapid access workup",
            "last_prediction": "lung_aca",
            "last_confidence": 97.4,
            "mrn": "MCC-88510",
            "doctor_id": "DR-206",
            "ward": "Thoracic Day Care",
            "case_summary": "New lung lesion with pleuritic pain referred through rapid access thoracic clinic.",
            "allergies": ["Penicillin"],
            "comorbidities": ["Asthma"],
            "medications": ["Budesonide-formoterol inhaler"],
            "history": [
                {"date": "2026-04-03", "event": "Rapid access referral", "detail": "Escalated due to progressive chest discomfort."},
                {"date": "2026-04-06", "event": "Biopsy accessioned", "detail": "Slides uploaded for pathology correlation."},
                {"date": "2026-04-07", "event": "Thoracic review", "detail": "Case prioritized for same-day AI review."},
            ],
        },
        {
            "id": "PT-1092",
            "name": "Aditi Chawla",
            "age": 55,
            "sex": "Female",
            "status": "Post-op oncology review",
            "clinic": "GI Oncology",
            "specimen": "Colon post-op section",
            "scheduled_time": "15:45",
            "risk_level": "Moderate",
            "diagnosis_stage": "Adjuvant pathway review",
            "last_prediction": "colon_aca",
            "last_confidence": 96.8,
            "mrn": "MCC-88534",
            "doctor_id": "DR-205",
            "ward": "GI Oncology Floor",
            "case_summary": "Post-operative colon cancer case being aligned to adjuvant planning and supportive care.",
            "allergies": ["None known"],
            "comorbidities": ["Vitamin D deficiency"],
            "medications": ["Calcium-vitamin D supplement"],
            "history": [
                {"date": "2026-03-30", "event": "Left colectomy done", "detail": "Resection specimen sent for staging review."},
                {"date": "2026-04-05", "event": "Recovery check", "detail": "Oral intake improving gradually."},
                {"date": "2026-04-07", "event": "Adjuvant planning visit", "detail": "Supportive medications and diet were discussed."},
            ],
        },
    ]
)

APPOINTMENTS = [
    {"id": "APT-5001", "patient_id": "PT-1042", "doctor_id": "DR-202", "type": "Thoracic consult", "time": "09:00", "status": "Checked in"},
    {"id": "APT-5002", "patient_id": "PT-1048", "doctor_id": "DR-203", "type": "Post-op oncology review", "time": "10:15", "status": "Waiting"},
    {"id": "APT-5003", "patient_id": "PT-1051", "doctor_id": "DR-201", "type": "Pathology correlation", "time": "11:30", "status": "Slides pending"},
    {"id": "APT-5004", "patient_id": "PT-1060", "doctor_id": "DR-204", "type": "Radiation planning", "time": "14:00", "status": "Confirmed"},
]

APPOINTMENTS.extend(
    [
        {"id": "APT-5005", "patient_id": "PT-1064", "doctor_id": "DR-205", "type": "Surgical oncology review", "time": "13:00", "status": "Confirmed"},
        {"id": "APT-5006", "patient_id": "PT-1068", "doctor_id": "DR-206", "type": "Thoracic urgent review", "time": "13:20", "status": "Priority"},
        {"id": "APT-5007", "patient_id": "PT-1072", "doctor_id": "DR-207", "type": "Supportive care review", "time": "13:45", "status": "Waiting"},
        {"id": "APT-5008", "patient_id": "PT-1076", "doctor_id": "DR-209", "type": "Medical oncology review", "time": "14:10", "status": "Confirmed"},
        {"id": "APT-5009", "patient_id": "PT-1080", "doctor_id": "DR-210", "type": "Pathology sign-out review", "time": "14:35", "status": "Slides ready"},
        {"id": "APT-5010", "patient_id": "PT-1084", "doctor_id": "DR-209", "type": "Pulmonary follow-up", "time": "15:00", "status": "Confirmed"},
        {"id": "APT-5011", "patient_id": "PT-1088", "doctor_id": "DR-206", "type": "Thoracic AI review", "time": "15:20", "status": "Priority"},
        {"id": "APT-5012", "patient_id": "PT-1092", "doctor_id": "DR-205", "type": "GI oncology review", "time": "15:45", "status": "Confirmed"},
    ]
)

PHARMACY_ITEMS = [
    {"name": "Pembrolizumab", "stock": 24, "status": "Available", "category": "Immunotherapy"},
    {"name": "Oxaliplatin", "stock": 11, "status": "Low stock", "category": "Chemotherapy"},
    {"name": "Ondansetron", "stock": 140, "status": "Available", "category": "Supportive care"},
    {"name": "Morphine Sulfate", "stock": 46, "status": "Controlled", "category": "Pain management"},
    {"name": "Pantoprazole", "stock": 78, "status": "Available", "category": "GI support"},
    {"name": "Tiotropium Inhaler", "stock": 34, "status": "Available", "category": "Pulmonary support"},
    {"name": "Dexamethasone", "stock": 52, "status": "Available", "category": "Supportive care"},
]

WARD_SUMMARY = [
    {"name": "Thoracic Day Care", "occupancy": "18/24", "nurse_in_charge": "N. D’Souza", "status": "Stable"},
    {"name": "GI Oncology Floor", "occupancy": "21/26", "nurse_in_charge": "R. Pillai", "status": "Busy"},
    {"name": "ICU Stepdown", "occupancy": "7/10", "nurse_in_charge": "A. Joseph", "status": "Monitored"},
    {"name": "Radiation Suite", "occupancy": "4/6", "nurse_in_charge": "S. Verma", "status": "On schedule"},
]

BILLING_ITEMS = [
    {"invoice": "INV-3001", "patient_id": "PT-1042", "service": "Biopsy pathology review", "amount": "₹18,500", "status": "Pending"},
    {"invoice": "INV-3002", "patient_id": "PT-1048", "service": "Post-op oncology consult", "amount": "₹12,000", "status": "Approved"},
    {"invoice": "INV-3003", "patient_id": "PT-1060", "service": "Chemotherapy planning", "amount": "₹26,800", "status": "Insurance review"},
]

REPORT_METRICS = [
    {"label": "Patients registered", "value": "128"},
    {"label": "Active treatment plans", "value": "42"},
    {"label": "Slides analyzed today", "value": "31"},
    {"label": "Tumor board cases", "value": "9"},
]

SYSTEM_ALERTS = [
    {"level": "Critical", "text": "2 patients require same-day pathology sign-out before tumor board."},
    {"level": "Info", "text": "4 pending slide reviews are awaiting consultant notes."},
    {"level": "Ops", "text": "Pharmacy flagged oxaliplatin as low stock."},
]

CARE_GUIDANCE = {
    "colon_aca": {
        "headline": "Colon adenocarcinoma pattern detected",
        "care_path": "Escalate to GI tumor board and correlate with resection margins and nodal status.",
        "medications": ["Review adjuvant chemotherapy pathway", "Ensure antiemetic support", "Assess thromboprophylaxis needs"],
        "next_steps": ["Finalize pathology sign-out", "Request MSI/MMR testing", "Schedule oncology follow-up within 72 hours"],
    },
    "colon_n": {
        "headline": "Benign colon tissue pattern detected",
        "care_path": "Correlate with endoscopy findings and review inflammatory or reactive changes.",
        "medications": ["Symptom-based GI support", "Avoid oncology-specific treatment until full correlation", "Review anti-inflammatory need"],
        "next_steps": ["Confirm benign correlation", "Document surveillance plan", "Return to GI follow-up"],
    },
    "lung_aca": {
        "headline": "Lung adenocarcinoma pattern detected",
        "care_path": "Escalate to thoracic oncology, correlate with imaging, and review molecular testing eligibility.",
        "medications": ["Prepare symptom support regimen", "Review bronchodilator support", "Assess antiemetic and pain-control plan"],
        "next_steps": ["Order molecular markers", "Present at thoracic MDT", "Plan staging workup"],
    },
    "lung_n": {
        "headline": "Benign lung tissue pattern detected",
        "care_path": "Correlate with inflammatory, infectious, and clinical context before discharging malignancy concern.",
        "medications": ["Maintain pulmonary symptom control", "No oncology treatment recommendation", "Review antibiotics only if indicated"],
        "next_steps": ["Issue benign pathology note", "Recommend interval imaging if needed", "Return to pulmonology follow-up"],
    },
    "lung_scc": {
        "headline": "Lung squamous cell carcinoma pattern detected",
        "care_path": "Escalate urgently to thoracic oncology and confirm with morphology plus immunohistochemistry.",
        "medications": ["Start symptom-directed supportive care planning", "Assess pain and cough regimen", "Review smoking cessation support"],
        "next_steps": ["Request P40/CK5-6 correlation if needed", "Stage with imaging", "Plan urgent oncology review"],
    },
}

PREDICTION_PROFILES = {
    "colon_aca": {
        "display": "Colon Adenocarcinoma",
        "cancer_label": "Active Colon Cancer",
        "triage_label": "GI Oncology Priority",
        "health_label": "Needs aggressive treatment planning",
        "diet_plan": "Soft high-protein colon support diet",
        "foods_to_eat": ["Curd rice", "Boiled vegetables", "Banana", "Oats", "Dal soup"],
        "foods_to_avoid": ["Processed meat", "Deep-fried food", "High-spice meals", "Alcohol"],
        "recommended_medicines": ["Antiemetic review", "Pain-control review", "GI protection review"],
        "wellness_score": "52 / 100",
        "follow_up": "Oncology review in 72 hours",
        "tone": "danger",
    },
    "colon_n": {
        "display": "Benign Colon Tissue",
        "cancer_label": "Benign / Observation",
        "triage_label": "Surveillance Pathway",
        "health_label": "Monitor and correlate clinically",
        "diet_plan": "High-fiber recovery diet",
        "foods_to_eat": ["Oatmeal", "Papaya", "Vegetable khichdi", "Curd", "Hydration"],
        "foods_to_avoid": ["Excessively spicy food", "Processed snacks", "Alcohol"],
        "recommended_medicines": ["Symptom-based GI support", "Hydration support"],
        "wellness_score": "84 / 100",
        "follow_up": "Routine GI follow-up",
        "tone": "success",
    },
    "lung_aca": {
        "display": "Lung Adenocarcinoma",
        "cancer_label": "Active Lung Cancer",
        "triage_label": "Thoracic Oncology Priority",
        "health_label": "Immediate staging and treatment planning",
        "diet_plan": "High-calorie lung recovery diet",
        "foods_to_eat": ["Protein shakes", "Egg whites", "Steamed fish", "Soft fruits", "Electrolyte fluids"],
        "foods_to_avoid": ["Smoking triggers", "Alcohol", "Excess salt", "Very oily food"],
        "recommended_medicines": ["Pain-control review", "Bronchodilator review", "Antiemetic review"],
        "wellness_score": "46 / 100",
        "follow_up": "Thoracic MDT within 48 hours",
        "tone": "danger",
    },
    "lung_n": {
        "display": "Benign Lung Tissue",
        "cancer_label": "Benign / Observation",
        "triage_label": "Pulmonary Follow-up",
        "health_label": "Stable, monitor symptoms",
        "diet_plan": "Pulmonary wellness diet",
        "foods_to_eat": ["Warm soups", "Leafy vegetables", "Seasonal fruits", "Hydration", "Light protein meals"],
        "foods_to_avoid": ["Smoking exposure", "Very cold beverages", "Heavy fried meals"],
        "recommended_medicines": ["Symptom inhaler review", "Supportive pulmonary care"],
        "wellness_score": "81 / 100",
        "follow_up": "Pulmonology review in 1 week",
        "tone": "success",
    },
    "lung_scc": {
        "display": "Lung Squamous Cell Carcinoma",
        "cancer_label": "Active Lung Cancer",
        "triage_label": "Urgent Thoracic Escalation",
        "health_label": "Needs urgent oncology escalation",
        "diet_plan": "High-calorie low-irritant lung diet",
        "foods_to_eat": ["Soft rice", "Paneer", "Moong dal", "Coconut water", "Stewed fruits"],
        "foods_to_avoid": ["Smoking exposure", "Very spicy food", "Alcohol", "Processed meat"],
        "recommended_medicines": ["Pain-control review", "Cough relief review", "Smoking cessation support"],
        "wellness_score": "43 / 100",
        "follow_up": "Urgent oncology review in 24 hours",
        "tone": "danger",
    },
}

CLINICAL_PLAYBOOK = {
    "colon_aca": {
        "nutrition_label": "Bowel-friendly protein support",
        "care_plan_label": "GI tumor board pathway",
        "disposition": "Adjuvant oncology planning",
        "watch_items": [
            "Monitor dehydration and bleeding risk",
            "Track pain score and bowel habit changes",
            "Review weight trend at every visit",
        ],
        "bundle_name": "Colon oncology support bundle",
        "ai_prescriptions": [
            {
                "medicine": "Ondansetron",
                "dose": "8 mg twice daily",
                "duration": "5 days",
                "instructions": "Use around treatment-review days for nausea support.",
            },
            {
                "medicine": "Pantoprazole",
                "dose": "40 mg once daily",
                "duration": "14 days",
                "instructions": "Take before breakfast for GI protection.",
            },
        ],
        "follow_up_type": "GI tumor board review",
        "follow_up_time": "2026-04-09 10:30",
    },
    "colon_n": {
        "nutrition_label": "Recovery and surveillance diet",
        "care_plan_label": "GI surveillance pathway",
        "disposition": "Outpatient observation",
        "watch_items": [
            "Review abdominal symptoms weekly",
            "Escalate only if bleeding or weight loss appears",
            "Correlate benign slides with endoscopy history",
        ],
        "bundle_name": "GI surveillance support bundle",
        "ai_prescriptions": [
            {
                "medicine": "Pantoprazole",
                "dose": "40 mg once daily",
                "duration": "7 days",
                "instructions": "Use only if reflux or gastritis symptoms persist.",
            },
        ],
        "follow_up_type": "GI surveillance review",
        "follow_up_time": "2026-04-11 09:15",
    },
    "lung_aca": {
        "nutrition_label": "Respiratory calorie support",
        "care_plan_label": "Thoracic MDT escalation",
        "disposition": "Thoracic oncology workup",
        "watch_items": [
            "Track oxygenation and cough burden",
            "Escalate new chest pain immediately",
            "Review appetite and hydration at each touchpoint",
        ],
        "bundle_name": "Thoracic adenocarcinoma support bundle",
        "ai_prescriptions": [
            {
                "medicine": "Tiotropium Inhaler",
                "dose": "1 inhalation daily",
                "duration": "14 days",
                "instructions": "Support dyspnea control while staging is completed.",
            },
            {
                "medicine": "Ondansetron",
                "dose": "8 mg twice daily",
                "duration": "5 days",
                "instructions": "Use if nausea develops during treatment workup.",
            },
        ],
        "follow_up_type": "Thoracic MDT review",
        "follow_up_time": "2026-04-09 08:45",
    },
    "lung_n": {
        "nutrition_label": "Pulmonary recovery support",
        "care_plan_label": "Pulmonary surveillance pathway",
        "disposition": "Pulmonology follow-up",
        "watch_items": [
            "Track cough, sputum, and fever history",
            "Repeat imaging if symptoms worsen",
            "Continue breathing exercise and hydration advice",
        ],
        "bundle_name": "Pulmonary surveillance support bundle",
        "ai_prescriptions": [
            {
                "medicine": "Tiotropium Inhaler",
                "dose": "1 inhalation daily",
                "duration": "7 days",
                "instructions": "Continue only if symptomatic and already clinically indicated.",
            },
        ],
        "follow_up_type": "Pulmonology follow-up",
        "follow_up_time": "2026-04-12 11:00",
    },
    "lung_scc": {
        "nutrition_label": "Urgent respiratory support diet",
        "care_plan_label": "Urgent thoracic escalation",
        "disposition": "High-priority oncology review",
        "watch_items": [
            "Monitor pain, cough, and hemoptysis closely",
            "Assess smoking cessation support urgently",
            "Escalate any respiratory distress same day",
        ],
        "bundle_name": "Thoracic SCC support bundle",
        "ai_prescriptions": [
            {
                "medicine": "Morphine Sulfate",
                "dose": "Low dose as needed",
                "duration": "3 days",
                "instructions": "Use only with consultant oversight for pain control.",
            },
            {
                "medicine": "Tiotropium Inhaler",
                "dose": "1 inhalation daily",
                "duration": "14 days",
                "instructions": "Support cough and breathing comfort where indicated.",
            },
        ],
        "follow_up_type": "Urgent thoracic oncology review",
        "follow_up_time": "2026-04-09 07:45",
    },
}


def get_doctor_map() -> Dict[str, Dict[str, Any]]:
    return {doctor["id"]: doctor for doctor in DOCTORS}


def get_patient_map() -> Dict[str, Dict[str, Any]]:
    return {patient["id"]: patient for patient in PATIENTS}


def get_patient(patient_id: str) -> Dict[str, Any]:
    return get_patient_map().get(patient_id, PATIENTS[0])


def get_doctor(doctor_id: str) -> Dict[str, Any]:
    return get_doctor_map().get(doctor_id, DOCTORS[0])


def build_initial_patient_state(patient: Dict[str, Any]) -> Dict[str, Any]:
    profile = PREDICTION_PROFILES[patient["last_prediction"]]
    playbook = CLINICAL_PLAYBOOK[patient["last_prediction"]]
    return {
        "ai_status": "Analyzed",
        "prediction_key": patient["last_prediction"],
        "prediction_label": profile["display"],
        "prediction_confidence": patient["last_confidence"],
        "cancer_label": profile["cancer_label"],
        "triage_label": profile["triage_label"],
        "health_label": profile["health_label"],
        "diet_plan": profile["diet_plan"],
        "foods_to_eat": list(profile["foods_to_eat"]),
        "foods_to_avoid": list(profile["foods_to_avoid"]),
        "recommended_medicines": list(profile["recommended_medicines"]),
        "doctor_prescriptions": [],
        "doctor_note": f"Awaiting consultant review. Baseline AI label: {profile['display']}.",
        "review_status": "Awaiting consultant sign-off",
        "follow_up": profile["follow_up"],
        "wellness_score": profile["wellness_score"],
        "tone": profile["tone"],
        "nutrition_label": playbook["nutrition_label"],
        "care_plan_label": playbook["care_plan_label"],
        "disposition": playbook["disposition"],
        "watch_items": list(playbook["watch_items"]),
        "care_bundle_status": f"{playbook['bundle_name']} available",
        "last_analysis_at": f"{TODAY} 09:00",
        "nutrition_note": f"Default AI diet plan assigned: {profile['diet_plan']}.",
    }


PATIENT_RUNTIME_STATE = {
    patient["id"]: build_initial_patient_state(patient) for patient in PATIENTS
}


def get_patient_state(patient_id: str) -> Dict[str, Any]:
    if patient_id not in PATIENT_RUNTIME_STATE:
        PATIENT_RUNTIME_STATE[patient_id] = build_initial_patient_state(get_patient(patient_id))
    return PATIENT_RUNTIME_STATE[patient_id]


def is_malignant(prediction_key: str) -> bool:
    return prediction_key not in {"colon_n", "lung_n"}


def patient_with_links(patient: Dict[str, Any]) -> Dict[str, Any]:
    doctor = get_doctor_map()[patient["doctor_id"]]
    state = get_patient_state(patient["id"])
    appointments = [appointment for appointment in APPOINTMENTS if appointment["patient_id"] == patient["id"]]
    enriched = dict(patient)
    enriched["doctor"] = doctor
    enriched.update(state)
    enriched["prescription_count"] = len(state["doctor_prescriptions"])
    enriched["appointment_count"] = len(appointments)
    if appointments:
        latest_appointment = appointments[-1]
        enriched["next_appointment"] = f"{latest_appointment['time']} · {latest_appointment['type']}"
    else:
        enriched["next_appointment"] = state["follow_up"]
    return enriched


def doctor_with_patients(doctor: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(doctor)
    patients = [patient_with_links(get_patient(pid)) for pid in doctor["assigned_patient_ids"]]
    appointments = [appointment for appointment in APPOINTMENTS if appointment["doctor_id"] == doctor["id"]]
    enriched["patients"] = patients
    enriched["active_cancer_cases"] = sum(1 for patient in patients if is_malignant(patient["prediction_key"]))
    enriched["patients_today"] = len(patients)
    enriched["pending_reviews"] = sum(1 for patient in patients if patient["review_status"] != "Reviewed and approved")
    enriched["critical_cases"] = sum(
        1 for patient in patients if patient["tone"] == "danger" and patient["review_status"] != "Reviewed and approved"
    )
    enriched["prescriptions_written"] = sum(patient["prescription_count"] for patient in patients)
    enriched["scheduled_followups"] = sum(
        1
        for appointment in appointments
        if any(keyword in appointment["type"].lower() for keyword in ("review", "follow", "board"))
    )
    return enriched


def build_hospital_overview() -> Dict[str, Any]:
    enriched_patients = [patient_with_links(patient) for patient in PATIENTS]
    return {
        "registered_patients": len(PATIENTS),
        "active_doctors": len(DOCTORS),
        "today_appointments": len(APPOINTMENTS),
        "critical_alerts": sum(1 for patient in enriched_patients if patient["tone"] == "danger"),
        "ai_completed": sum(1 for patient in enriched_patients if patient["ai_status"] == "Analyzed"),
        "tumor_board_cases": sum(1 for patient in enriched_patients if "tumor board" in patient["review_status"].lower()),
        "prescriptions_written": sum(patient["prescription_count"] for patient in enriched_patients),
        "followups_scheduled": sum(
            1
            for appointment in APPOINTMENTS
            if any(keyword in appointment["type"].lower() for keyword in ("review", "follow", "board"))
        ),
    }


def build_navigation() -> List[Dict[str, str]]:
    return [
        {"label": "Overview", "endpoint": "home"},
        {"label": "Patients", "endpoint": "patients"},
        {"label": "Doctors", "endpoint": "doctors"},
        {"label": "Appointments", "endpoint": "appointments"},
        {"label": "Pathology AI", "endpoint": "pathology_ai"},
        {"label": "Pharmacy", "endpoint": "pharmacy"},
        {"label": "Wards", "endpoint": "wards"},
        {"label": "Billing", "endpoint": "billing"},
        {"label": "Reports", "endpoint": "reports"},
        {"label": "Admin", "endpoint": "admin"},
    ]


def pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def append_patient_history(patient_id: str, event: str, detail: str, date: str = TODAY) -> None:
    patient = get_patient(patient_id)
    history = patient.setdefault("history", [])
    history.insert(0, {"date": date, "event": event, "detail": detail})


def inventory_status_for(item: Dict[str, Any]) -> str:
    stock = int(item["stock"])
    if stock <= 0:
        return "Out of stock"
    if stock <= 12:
        return "Low stock"
    if item["category"] == "Pain management":
        return "Controlled"
    return "Available"


def find_inventory_item(medicine_name: str) -> Dict[str, Any] | None:
    normalized = medicine_name.lower().strip()
    for item in PHARMACY_ITEMS:
        item_name = item["name"].lower()
        if normalized == item_name or normalized in item_name or item_name in normalized:
            return item
    return None


def allocate_inventory_stock(medicine_name: str) -> None:
    item = find_inventory_item(medicine_name)
    if item is None:
        return
    item["stock"] = max(0, int(item["stock"]) - 1)
    item["status"] = inventory_status_for(item)


def build_pharmacy_snapshot() -> List[Dict[str, Any]]:
    allocations: Dict[str, int] = {item["name"]: 0 for item in PHARMACY_ITEMS}
    for patient in PATIENTS:
        for prescription in get_patient_state(patient["id"])["doctor_prescriptions"]:
            item = find_inventory_item(prescription["medicine"])
            if item is not None:
                allocations[item["name"]] += 1

    snapshot = []
    for item in PHARMACY_ITEMS:
        snapshot.append(
            {
                **item,
                "allocated": allocations[item["name"]],
            }
        )
    return snapshot


def schedule_patient_follow_up(patient_id: str, doctor_id: str, appointment_type: str, slot: str, status: str = "Planned") -> None:
    for appointment in APPOINTMENTS:
        if appointment["patient_id"] == patient_id and appointment["type"] == appointment_type and appointment["time"] == slot:
            return

    APPOINTMENTS.append(
        {
            "id": f"APT-{5000 + len(APPOINTMENTS) + 1}",
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "type": appointment_type,
            "time": slot,
            "status": status,
        }
    )


def get_next_patient_for_doctor(doctor_id: str, current_patient_id: str) -> Dict[str, Any] | None:
    assigned_ids = get_doctor(doctor_id)["assigned_patient_ids"]
    if current_patient_id not in assigned_ids or len(assigned_ids) < 2:
        return None
    current_index = assigned_ids.index(current_patient_id)
    next_id = assigned_ids[(current_index + 1) % len(assigned_ids)]
    return patient_with_links(get_patient(next_id))


def build_system_alerts() -> List[Dict[str, str]]:
    patients = [patient_with_links(patient) for patient in PATIENTS]
    urgent = [patient for patient in patients if patient["tone"] == "danger" and patient["review_status"] != "Reviewed and approved"]
    tumor_board = [patient for patient in patients if "tumor board" in patient["review_status"].lower()]
    low_stock = [item["name"] for item in PHARMACY_ITEMS if item["status"] in {"Low stock", "Out of stock"}]
    without_prescriptions = [patient for patient in urgent if patient["prescription_count"] == 0]

    alerts: List[Dict[str, str]] = []
    if urgent:
        alerts.append({"level": "Critical", "text": f"{len(urgent)} high-risk cancer cases still need consultant closure."})
    if tumor_board:
        alerts.append({"level": "Board", "text": f"{len(tumor_board)} patients are queued for tumor board discussion."})
    if low_stock:
        alerts.append({"level": "Ops", "text": f"Pharmacy attention: {', '.join(low_stock)} stock requires review."})
    if without_prescriptions:
        alerts.append({"level": "Care", "text": f"{len(without_prescriptions)} urgent patients do not yet have a drafted prescription set."})

    return alerts or SYSTEM_ALERTS


def build_report_metrics() -> List[Dict[str, str]]:
    patients = [patient_with_links(patient) for patient in PATIENTS]
    return [
        {"label": "Patients registered", "value": str(len(PATIENTS))},
        {"label": "Active treatment plans", "value": str(sum(1 for patient in patients if is_malignant(patient["prediction_key"])))},
        {"label": "Slides analyzed today", "value": str(sum(1 for patient in patients if patient["ai_status"] == "Analyzed"))},
        {"label": "Tumor board cases", "value": str(sum(1 for patient in patients if "tumor board" in patient["review_status"].lower()))},
        {"label": "Prescriptions drafted", "value": str(sum(patient["prescription_count"] for patient in patients))},
        {
            "label": "Follow-ups scheduled",
            "value": str(
                sum(
                    1
                    for appointment in APPOINTMENTS
                    if any(keyword in appointment["type"].lower() for keyword in ("review", "follow", "board"))
                )
            ),
        },
    ]


def build_analysis_summary(predicted_key: str, confidence: float) -> Dict[str, Any]:
    guidance = CARE_GUIDANCE.get(predicted_key, CARE_GUIDANCE["lung_n"])
    profile = PREDICTION_PROFILES.get(predicted_key, PREDICTION_PROFILES["lung_n"])
    playbook = CLINICAL_PLAYBOOK.get(predicted_key, CLINICAL_PLAYBOOK["lung_n"])
    return {
        "headline": guidance["headline"],
        "care_path": guidance["care_path"],
        "recommended_medications": guidance["medications"],
        "next_steps": guidance["next_steps"],
        "doctor_note": (
            f"AI fused prediction favored {profile['display']} at {confidence}%. "
            "Correlate with morphology, clinical notes, and prior pathology before final sign-out."
        ),
        "diet_plan": profile["diet_plan"],
        "foods_to_eat": profile["foods_to_eat"],
        "foods_to_avoid": profile["foods_to_avoid"],
        "health_label": profile["health_label"],
        "care_plan_label": playbook["care_plan_label"],
        "nutrition_label": playbook["nutrition_label"],
        "disposition": playbook["disposition"],
        "watch_items": playbook["watch_items"],
        "bundle_name": playbook["bundle_name"],
    }


def apply_prediction_to_patient(patient_id: str, prediction_key: str, confidence: float) -> None:
    state = get_patient_state(patient_id)
    profile = PREDICTION_PROFILES[prediction_key]
    playbook = CLINICAL_PLAYBOOK[prediction_key]
    state.update(
        {
            "ai_status": "Analyzed",
            "prediction_key": prediction_key,
            "prediction_label": profile["display"],
            "prediction_confidence": round(confidence, 2),
            "cancer_label": profile["cancer_label"],
            "triage_label": profile["triage_label"],
            "health_label": profile["health_label"],
            "diet_plan": profile["diet_plan"],
            "foods_to_eat": list(profile["foods_to_eat"]),
            "foods_to_avoid": list(profile["foods_to_avoid"]),
            "recommended_medicines": list(profile["recommended_medicines"]),
            "review_status": "AI ready for doctor review",
            "follow_up": profile["follow_up"],
            "wellness_score": profile["wellness_score"],
            "tone": profile["tone"],
            "nutrition_label": playbook["nutrition_label"],
            "care_plan_label": playbook["care_plan_label"],
            "disposition": playbook["disposition"],
            "watch_items": list(playbook["watch_items"]),
            "care_bundle_status": f"{playbook['bundle_name']} available",
            "last_analysis_at": f"{TODAY} 11:30",
            "nutrition_note": f"AI updated diet plan to {profile['diet_plan']}.",
        }
    )
    append_patient_history(
        patient_id,
        "AI pathology analysis updated",
        f"Fused model favored {profile['display']} at {round(confidence, 2)}% confidence.",
    )


def normalize_csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_page_message() -> str | None:
    return request.args.get("message")


def get_modal_name() -> str | None:
    return request.args.get("modal")


def get_model_service_status(silent: bool = True) -> Dict[str, Any] | None:
    try:
        return fetch_model_service_health()
    except ModelServiceError:
        if silent:
            return None
        raise


def build_model_service_context() -> Dict[str, Any]:
    health = get_model_service_status(silent=True)
    artifacts = health.get("artifacts", {}) if health else {}
    return {
        "model_service_url": get_model_api_url(),
        "model_service_reachable": bool(health and health.get("status") == "ok"),
        "model_ready": bool(artifacts.get("fusion", False)),
        "convnext_ready": bool(artifacts.get("convnext", False)),
        "phikon_ready": bool(artifacts.get("phikon", False)),
        "model_service_health": health,
    }


def metrics_route_label() -> str:
    if request.url_rule is not None and request.url_rule.rule:
        return request.url_rule.rule
    return request.path


def should_skip_metrics(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in METRICS_EXCLUDED_PATH_PREFIXES)


def update_runtime_metrics() -> None:
    health = get_model_service_status(silent=True)
    artifacts = health.get("artifacts", {}) if health else {}
    device_info = health.get("device", {}) if health else {}

    MODEL_SERVICE_AVAILABLE.set(1 if health else 0)
    MODEL_ARTIFACT_READY.labels(artifact="fusion").set(1 if artifacts.get("fusion") else 0)
    MODEL_ARTIFACT_READY.labels(artifact="convnext").set(1 if artifacts.get("convnext") else 0)
    MODEL_ARTIFACT_READY.labels(artifact="phikon").set(1 if artifacts.get("phikon") else 0)

    GPU_AVAILABLE.labels(backend="cuda").set(1 if device_info.get("cuda") else 0)
    GPU_AVAILABLE.labels(backend="mps").set(1 if device_info.get("mps") else 0)
    GPU_DEVICE_COUNT.set(int(device_info.get("cuda_device_count", 0) or 0))
    GPU_MEMORY_ALLOCATED_BYTES.set(0)
    GPU_MEMORY_RESERVED_BYTES.set(0)


def initialize_metrics() -> None:
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith("/static/"):
            continue
        methods = sorted(method for method in rule.methods if method in {"GET", "POST"})
        for method in methods:
            HTTP_REQUESTS_TOTAL.labels(method=method, route=rule.rule, status_code="200")
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=rule.rule)

    for stage in ("upload", "prediction", "gradcam"):
        for status in ("success", "error"):
            INFERENCE_TOTAL.labels(stage=stage, status=status)

    MODEL_ARTIFACT_READY.labels(artifact="fusion")
    MODEL_ARTIFACT_READY.labels(artifact="convnext")
    MODEL_ARTIFACT_READY.labels(artifact="phikon")
    GPU_AVAILABLE.labels(backend="cuda")
    GPU_AVAILABLE.labels(backend="mps")


@app.context_processor
def inject_global_context() -> Dict[str, Any]:
    return {
        "app_name": APP_NAME,
        "navigation": build_navigation(),
    }


@app.before_request
def start_request_timer() -> None:
    if should_skip_metrics(request.path):
        return
    g.request_started_at = time.perf_counter()


@app.after_request
def observe_request_metrics(response: Response) -> Response:
    if should_skip_metrics(request.path):
        return response

    started_at = getattr(g, "request_started_at", None)
    if started_at is None:
        return response

    route = metrics_route_label()
    duration = time.perf_counter() - started_at
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        route=route,
        status_code=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        route=route,
    ).observe(duration)
    return response


@app.route("/patients/<patient_id>/manage", methods=["POST"])
def manage_patient(patient_id: str):
    state = get_patient_state(patient_id)
    patient = get_patient(patient_id)
    action = request.form.get("action", "")
    return_to = request.form.get("return_to", "patient_detail")
    if return_to not in {"patient_detail", "pathology_ai"}:
        return_to = "patient_detail"

    message = "No change was made."

    if action == "prescribe":
        medicine = request.form.get("medicine", "").strip()
        dose = request.form.get("dose", "").strip()
        duration = request.form.get("duration", "").strip()
        instructions = request.form.get("instructions", "").strip()
        if medicine:
            state["doctor_prescriptions"].append(
                {
                    "medicine": medicine,
                    "dose": dose or "To be reviewed",
                    "duration": duration or "As directed",
                    "instructions": instructions or "Take as prescribed by consultant",
                }
            )
            allocate_inventory_stock(medicine)
            state["review_status"] = "Prescription drafted by consultant"
            state["care_bundle_status"] = "Custom prescription plan active"
            append_patient_history(
                patient_id,
                "Prescription drafted",
                f"{medicine} added with consultant instructions for {patient['name']}.",
            )
            message = f"Prescription added for {patient['name']}."

    elif action == "assign_diet":
        diet_name = request.form.get("diet_name", "").strip()
        foods_to_eat = normalize_csv_items(request.form.get("foods_to_eat", ""))
        foods_to_avoid = normalize_csv_items(request.form.get("foods_to_avoid", ""))
        diet_note = request.form.get("diet_note", "").strip()
        if diet_name:
            state["diet_plan"] = diet_name
        if foods_to_eat:
            state["foods_to_eat"] = foods_to_eat
        if foods_to_avoid:
            state["foods_to_avoid"] = foods_to_avoid
        if diet_note:
            state["nutrition_note"] = diet_note
        state["review_status"] = "Diet plan assigned"
        append_patient_history(
            patient_id,
            "Diet plan updated",
            f"Consultant assigned diet plan: {state['diet_plan']}.",
        )
        message = f"Diet plan updated for {patient['name']}."

    elif action == "save_note":
        doctor_note = request.form.get("doctor_note", "").strip()
        if doctor_note:
            state["doctor_note"] = doctor_note
            state["review_status"] = "Doctor note saved"
            append_patient_history(
                patient_id,
                "Doctor note updated",
                "Consultant note added to patient care record.",
            )
            message = f"Doctor note saved for {patient['name']}."

    elif action == "apply_ai_bundle":
        playbook = CLINICAL_PLAYBOOK[state["prediction_key"]]
        existing_medicines = {item["medicine"].lower() for item in state["doctor_prescriptions"]}
        added_count = 0
        for order in playbook["ai_prescriptions"]:
            if order["medicine"].lower() in existing_medicines:
                continue
            state["doctor_prescriptions"].append(dict(order))
            allocate_inventory_stock(order["medicine"])
            added_count += 1
        state["review_status"] = "AI care bundle applied"
        state["care_bundle_status"] = f"{playbook['bundle_name']} applied"
        state["follow_up"] = f"{playbook['follow_up_type']} · {playbook['follow_up_time']}"
        state["doctor_note"] = (
            f"{playbook['bundle_name']} activated. Review AI-supported orders and correlate with clinical findings before final sign-off."
        )
        schedule_patient_follow_up(
            patient_id,
            patient["doctor_id"],
            playbook["follow_up_type"],
            playbook["follow_up_time"],
        )
        append_patient_history(
            patient_id,
            "AI care bundle applied",
            f"{playbook['bundle_name']} activated with {max(added_count, 0)} supportive medication orders.",
        )
        message = f"AI care bundle applied for {patient['name']}."

    elif action == "schedule_followup":
        follow_up_type = request.form.get("follow_up_type", "").strip()
        follow_up_time = request.form.get("follow_up_time", "").strip()
        if follow_up_type and follow_up_time:
            schedule_patient_follow_up(
                patient_id,
                patient["doctor_id"],
                follow_up_type,
                follow_up_time,
            )
            state["follow_up"] = f"{follow_up_type} · {follow_up_time}"
            state["review_status"] = "Follow-up scheduled"
            append_patient_history(
                patient_id,
                "Follow-up scheduled",
                f"{follow_up_type} booked for {follow_up_time}.",
            )
            message = f"Follow-up scheduled for {patient['name']}."

    elif action == "mark_reviewed":
        state["review_status"] = "Reviewed and approved"
        state["care_bundle_status"] = "Consultant-approved plan active"
        append_patient_history(
            patient_id,
            "Case reviewed",
            "Consultant signed off current AI analysis and treatment plan.",
        )
        message = f"{patient['name']} marked as reviewed."

    elif action == "tumor_board":
        state["review_status"] = "Sent to tumor board"
        state["care_bundle_status"] = "Tumor board review pending"
        schedule_patient_follow_up(
            patient_id,
            patient["doctor_id"],
            "Tumor board discussion",
            "2026-04-09 16:00",
            status="Confirmed",
        )
        append_patient_history(
            patient_id,
            "Tumor board escalation",
            "Case escalated for multidisciplinary tumor board discussion.",
        )
        message = f"{patient['name']} escalated to tumor board."

    return redirect(url_for(return_to, patient_id=patient_id, message=message))


@app.route("/")
def home() -> str:
    enriched_patients = [patient_with_links(patient) for patient in PATIENTS]
    return render_template(
        "home.html",
        active_page="home",
        page_title="Hospital Overview",
        page_intro="A command center for patients, doctors, oncology workflows, pathology AI, and hospital operations.",
        overview=build_hospital_overview(),
        alerts=build_system_alerts(),
        patients=enriched_patients,
        doctors=[doctor_with_patients(doctor) for doctor in DOCTORS],
        lead_doctor=doctor_with_patients(get_doctor("DR-201")),
        page_message=get_page_message(),
    )


@app.route("/healthz")
def healthz() -> Dict[str, Any]:
    update_runtime_metrics()
    model_health = get_model_service_status(silent=True)
    return {
        "status": "ok" if model_health else "degraded",
        "app": APP_NAME,
        "model_service_url": get_model_api_url(),
        "model_service": model_health,
    }


@app.route("/metrics")
def metrics() -> Response:
    update_runtime_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/patients")
def patients() -> str:
    return render_template(
        "patients.html",
        active_page="patients",
        page_title="Patient Dashboard",
        page_intro="Browse all oncology patients, current cancer labels, AI status, diet plans, and prescription progress.",
        patients=[patient_with_links(patient) for patient in PATIENTS],
        page_message=get_page_message(),
    )


@app.route("/patients/<patient_id>")
def patient_detail(patient_id: str) -> str:
    patient = patient_with_links(get_patient(patient_id))
    guidance = build_analysis_summary(patient["prediction_key"], patient["prediction_confidence"])
    return render_template(
        "patient_detail.html",
        active_page="patients",
        page_title=patient["name"],
        page_intro="Patient profile, AI cancer label, diet plan, prescriptions, and care workflow.",
        patient=patient,
        care_guidance=guidance,
        appointments=[appointment for appointment in APPOINTMENTS if appointment["patient_id"] == patient["id"]],
        next_patient=get_next_patient_for_doctor(patient["doctor"]["id"], patient["id"]),
        page_message=get_page_message(),
        open_modal=get_modal_name(),
    )


@app.route("/doctors")
def doctors() -> str:
    return render_template(
        "doctors.html",
        active_page="doctors",
        page_title="Doctor Dashboard",
        page_intro="Review specialist workload, assigned patients, active cancer cases, and current oncology responsibilities.",
        doctors=[doctor_with_patients(doctor) for doctor in DOCTORS],
        page_message=get_page_message(),
    )


@app.route("/doctors/<doctor_id>")
def doctor_detail(doctor_id: str) -> str:
    doctor = doctor_with_patients(get_doctor(doctor_id))
    return render_template(
        "doctor_detail.html",
        active_page="doctors",
        page_title=doctor["name"],
        page_intro="Assigned patients, workload metrics, contact details, and live case status for this consultant.",
        doctor=doctor,
        page_message=get_page_message(),
    )


@app.route("/appointments")
def appointments() -> str:
    doctor_map = get_doctor_map()
    patient_map = get_patient_map()
    enriched = []
    for appointment in APPOINTMENTS:
        patient = patient_with_links(patient_map[appointment["patient_id"]])
        enriched.append(
            {
                **appointment,
                "patient": patient,
                "doctor": doctor_map[appointment["doctor_id"]],
            }
        )
    return render_template(
        "appointments.html",
        active_page="appointments",
        page_title="Appointments",
        page_intro="Track consults, pathology reviews, treatment planning, and day-care scheduling.",
        appointments=enriched,
        page_message=get_page_message(),
    )


@app.route("/pathology-ai", methods=["GET", "POST"])
def pathology_ai() -> str:
    requested_patient_id = request.values.get("patient_id", PATIENTS[0]["id"])
    selected_patient = patient_with_links(get_patient(requested_patient_id))
    analysis_summary = build_analysis_summary(
        selected_patient["prediction_key"],
        selected_patient["prediction_confidence"],
    )
    model_context = build_model_service_context()
    context: Dict[str, Any] = {
        "active_page": "pathology_ai",
        "page_title": "Pathology AI Analysis",
        "page_intro": "Upload a histopathology slide for fused cancer prediction and ConvNeXt Grad-CAM explainability.",
        "patients": [patient_with_links(patient) for patient in PATIENTS],
        "selected_patient": selected_patient,
        "result": None,
        "error": None,
        "xai_error": None,
        "uploaded_image": None,
        "gradcam_image": None,
        "analysis_summary": analysis_summary,
        "page_message": get_page_message(),
    }
    context.update(model_context)

    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or uploaded.filename == "":
            context["error"] = "Please upload a histopathology image before running detection."
            INFERENCE_TOTAL.labels(stage="upload", status="error").inc()
            return render_template("pathology_ai.html", **context)

        try:
            image_bytes = uploaded.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            context["uploaded_image"] = pil_to_data_url(image, fmt="PNG")
            prediction_started_at = time.perf_counter()
            prediction = predict_with_model_service(
                image_bytes=image_bytes,
                filename=uploaded.filename or f"{requested_patient_id}.png",
                include_gradcam=True,
            )
            INFERENCE_DURATION_SECONDS.labels(stage="prediction").observe(
                time.perf_counter() - prediction_started_at
            )
            INFERENCE_TOTAL.labels(stage="prediction", status="success").inc()
            context["model_service_reachable"] = True
            context["result"] = {
                "prediction": prediction["prediction"],
                "class_probabilities": prediction["class_probabilities"],
                "model_breakdown": prediction["model_breakdown"],
                "top_predictions": prediction["top_predictions"],
            }
            apply_prediction_to_patient(
                requested_patient_id,
                prediction["prediction"]["class_key"],
                prediction["prediction"]["confidence"],
            )
            refreshed_patient = patient_with_links(get_patient(requested_patient_id))
            context["selected_patient"] = refreshed_patient
            context["patients"] = [patient_with_links(patient) for patient in PATIENTS]
            context["analysis_summary"] = build_analysis_summary(
                refreshed_patient["prediction_key"],
                refreshed_patient["prediction_confidence"],
            )
            if prediction.get("gradcam_image_data_url"):
                gradcam_seconds = ((prediction.get("timings") or {}).get("gradcam_seconds"))
                if gradcam_seconds is not None:
                    INFERENCE_DURATION_SECONDS.labels(stage="gradcam").observe(float(gradcam_seconds))
                INFERENCE_TOTAL.labels(stage="gradcam", status="success").inc()
                context["gradcam_image"] = prediction["gradcam_image_data_url"]
            elif prediction.get("xai_error"):
                INFERENCE_TOTAL.labels(stage="gradcam", status="error").inc()
                context["xai_error"] = prediction["xai_error"]
        except Exception as exc:
            INFERENCE_TOTAL.labels(stage="prediction", status="error").inc()
            context["error"] = str(exc)

    return render_template("pathology_ai.html", **context)


@app.route("/pharmacy")
def pharmacy() -> str:
    return render_template(
        "pharmacy.html",
        active_page="pharmacy",
        page_title="Pharmacy Management",
        page_intro="Monitor oncology medication stock, supportive drugs, and controlled inventory.",
        items=build_pharmacy_snapshot(),
        page_message=get_page_message(),
    )


@app.route("/wards")
def wards() -> str:
    return render_template(
        "wards.html",
        active_page="wards",
        page_title="Ward Management",
        page_intro="Track occupancy, unit readiness, nursing ownership, and inpatient flow.",
        wards=WARD_SUMMARY,
        patients=[patient_with_links(patient) for patient in PATIENTS],
        page_message=get_page_message(),
    )


@app.route("/billing")
def billing() -> str:
    patient_map = get_patient_map()
    enriched = [{**item, "patient": patient_with_links(patient_map[item["patient_id"]])} for item in BILLING_ITEMS]
    return render_template(
        "billing.html",
        active_page="billing",
        page_title="Billing and Insurance",
        page_intro="Review invoices, insurance status, and oncology service charges.",
        items=enriched,
        page_message=get_page_message(),
    )


@app.route("/reports")
def reports() -> str:
    return render_template(
        "reports.html",
        active_page="reports",
        page_title="Clinical Reports",
        page_intro="Operational KPIs, pathology throughput, multidisciplinary workload, and AI adoption summaries.",
        metrics=build_report_metrics(),
        alerts=build_system_alerts(),
        patients=[patient_with_links(patient) for patient in PATIENTS],
        page_message=get_page_message(),
    )


@app.route("/admin")
def admin() -> str:
    return render_template(
        "admin.html",
        active_page="admin",
        page_title="Administration",
        page_intro="Manage hospital systems, staffing, escalations, audit readiness, and platform health.",
        doctors=[doctor_with_patients(doctor) for doctor in DOCTORS],
        alerts=build_system_alerts(),
        wards=WARD_SUMMARY,
        page_message=get_page_message(),
    )


initialize_metrics()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
