"""
test_cases.py — 10 test cases from the [Sovra] LLM Testcase PDF.

Each test case provides the full input payload for POST /explain-defect
and the expected recommended_action from the LLM.

Run:
    python test_cases.py                    # Run all 10 tests against the API
    python test_cases.py --test TC-001      # Run a single test
    python test_cases.py --dry-run          # Print payloads without calling API
"""

import argparse
import json
import sys
import time

import httpx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_URL = "http://localhost:8000/explain-defect"

# ---------------------------------------------------------------------------
# Test Case Definitions
# ---------------------------------------------------------------------------

TEST_CASES = {
    "TC-001": {
        "description": "Recurring BMFO at AOI-04 — should recommend clean_station",
        "expected_action": "clean_station",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
                "factory_type": "automotive electronics manufacturing",
                "region": "Germany",
                "branch": "Berlin",
                "quality_standard": "OEM-QA-Level-2",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "production_line_name": "PCB Assembly Line A3",
                "line_function": "post-etching PCB inspection",
                "recent_yield_7d": "96.8%",
                "average_daily_volume": "3200_boards",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
                "station_function": "visual defect inspection after etching",
                "machine_id": "AOI-MACHINE-04",
                "camera_id": "CAM-TOP-01",
                "camera_view": "top_camera",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "PCB used in vehicle body control and lighting control electronics",
                "affected_system": "body_control",
                "subsystem": "lighting_control",
                "safety_criticality": "medium",
                "revision": "Rev-B",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_name": "Demo PCB Supplier",
                "supplied_material": "FR-4 copper-clad laminate",
                "supplier_quality_status": "normal",
                "incoming_inspection_status": "passed",
                "supplier_issue_found": False,
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
                "lot_id": "LOT-PCB-X100-044",
                "work_order_id": "WO-2026-8812",
                "batch_size": "1800_boards",
                "batch_status": "in_progress",
                "quarantine_status": "false",
            },
            "defect_history": {
                "defect_code": "BMFO",
                "same_defect_count_30d": 9,
                "same_defect_count_90d": 21,
                "scope": "same_factory_same_line_same_station_same_product",
                "trend": "increasing",
                "most_common_shift": "Afternoon",
                "recurring_location": "center_or_large_crop_area",
                "same_location_count_30d": 6,
            },
            "repair_records": {
                "related_repair_count_30d": 3,
                "common_repair_action": "cleaned affected board and reinspected",
                "repair_result": "3 cleaned_and_passed",
            },
            "maintenance_logs": {
                "latest_maintenance_id": "MNT-2026-00502",
                "maintenance_type": "routine_lens_cleaning_and_lighting_inspection",
                "maintenance_date": "2026-05-02",
                "technician_note": "minor dust accumulation near board loading area",
                "recommendation": "full cleaning if foreign-object detections continue",
            },
            "final_dispositions": {
                "last_5_similar_cases": 5,
                "cleaned_and_passed": 3,
                "manual_review_pending": 2,
                "rejected": 0,
                "rework_failed": 0,
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 7,
                "estimated_defect_regions": 4,
                "number_of_defects": 4,
                "highest_confidence": 0.9245,
                "highest_severity_hint": "medium",
                "overall_status_hint": "manual_review_recommended",
                "primary_defect_code": "BMFO",
                "primary_confidence": 0.9245,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
    "TC-002": {
        "description": "Recurring OP in functional zone, repair failed — should escalate_engineering",
        "expected_action": "escalate_engineering",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
                "factory_type": "automotive electronics manufacturing",
                "region": "Germany",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "production_line_name": "PCB Assembly Line A3",
                "line_function": "post-etching PCB inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
                "machine_id": "AOI-MACHINE-04",
                "camera_id": "CAM-TOP-01",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "PCB used for body control and power distribution signal paths",
                "affected_system": "body_control",
                "subsystem": "power_distribution_signal_path",
                "safety_criticality": "high",
                "revision": "Rev-B",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_name": "Demo PCB Supplier",
                "supplier_quality_status": "normal",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
                "lot_id": "LOT-PCB-X100-044",
                "batch_status": "in_progress",
            },
            "defect_history": {
                "defect_code": "OP",
                "same_defect_count_90d": 12,
                "coordinate_region": "center_and_lower_center_crop_regions",
                "functional_zone": True,
                "functional_zone_description": "BCM-01 power distribution conductor path",
            },
            "repair_records": {
                "repair_id": "RPR-2026-00421",
                "repair_action": "manual copper bridge repair",
                "repair_verification": "passed continuity test",
                "recurrence_after_repair": True,
                "recurrence_after_repair_days": 6,
            },
            "maintenance_logs": {},
            "final_dispositions": {
                "current_case_disposition": "pending_engineering_review",
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 12,
                "estimated_defect_regions": 7,
                "number_of_defects": 7,
                "highest_confidence": 0.9512,
                "highest_severity_hint": "high",
                "overall_status_hint": "critical_defect_detected",
                "primary_defect_code": "OP",
                "primary_confidence": 0.9512,
                "secondary_defect_code": "SC",
                "secondary_confidence": 0.42,
                "detections": [],
            },
        },
    },
    "TC-003": {
        "description": "SC after calibration, 52% manual confirmation — should recommend calibration_review",
        "expected_action": "calibration_review",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
                "branch": "Berlin",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "post-etching PCB inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
                "machine_id": "AOI-MACHINE-04",
                "camera_id": "CAM-TOP-01",
                "calibration_id": "CAL-2026-05-15",
                "calibration_date": "2026-05-15",
                "calibration_type": "camera_and_lighting_recalibration",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "automotive control PCB with reflective copper trace regions",
                "surface_finish": "standard_copper_trace_finish",
                "visual_sensitivity": "high_reflective_copper_regions",
                "revision": "Rev-B",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_quality_status": "normal",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
                "batch_status": "in_progress",
            },
            "defect_history": {
                "defect_code": "SC",
                "sc_detection_rate_before_calibration": "4.2_per_1000_boards",
                "sc_detection_rate_after_calibration": "5.8_per_1000_boards",
                "manual_review_confirmation_rate": "52_percent",
            },
            "repair_records": {},
            "maintenance_logs": {
                "maintenance_event": "AOI_camera_and_lighting_recalibration",
                "calibration_monitoring_window": "active",
            },
            "final_dispositions": {
                "current_case_disposition": "manual_review_required_after_calibration_drift",
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 1,
                "estimated_defect_regions": 1,
                "number_of_defects": 1,
                "highest_confidence": 0.8821,
                "highest_severity_hint": "medium",
                "overall_status_hint": "manual_review_recommended",
                "primary_defect_code": "SC",
                "primary_confidence": 0.8821,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
    "TC-004": {
        "description": "Recurring CS after robot repair — should escalate_engineering",
        "expected_action": "escalate_engineering",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_type": "automotive electronics manufacturing",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "PCB handling and inspection flow",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
                "linked_machine": "HANDLING-ROBOT-02",
                "linked_machine_function": "automated PCB transfer near conductor-sensitive area",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "automotive body control PCB with fine conductor traces near the handling path",
                "critical_features": "fine conductor traces near handling path",
                "safety_criticality": "medium_high",
                "revision": "Rev-B",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_quality_status": "normal",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
            },
            "defect_history": {
                "defect_code": "CS",
                "same_defect_count_30d": 5,
                "same_defect_count_90d": 5,
                "scope": "same_station_same_product",
            },
            "repair_records": {
                "repair_id": "RPR-2026-00512",
                "repair_action": "polished contact surface and replaced handling guide",
                "repair_verification": "visual_check_passed",
                "recurrence_after_repair": True,
                "recurrence_after_repair_days": 6,
            },
            "maintenance_logs": {},
            "final_dispositions": {
                "current_case_disposition": "manual_review_pending",
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 13,
                "estimated_defect_regions": 5,
                "number_of_defects": 5,
                "highest_confidence": 0.9698,
                "highest_severity_hint": "high",
                "overall_status_hint": "critical_defect_detected",
                "primary_defect_code": "CS",
                "primary_confidence": 0.9698,
                "secondary_defect_code": "CFO",
                "secondary_confidence": 0.78,
                "detections": [],
            },
        },
    },
    "TC-005": {
        "description": "Rev-C qualification, no CV defect — should pass or log_and_monitor",
        "expected_action": "pass",
        "alt_actions": ["log_and_monitor"],
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "qualification batch inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-C",
                "product_name": "Main Control PCB",
                "revision": "Rev-C",
                "product_description": "updated material revision of PCB-X100",
                "qualification_status": "new_supplier_qualification",
            },
            "supplier": {
                "supplier_id": "SUP-077",
                "supplier_name": "New Substrate Supplier",
                "supplier_quality_status": "new_qualification_supplier",
            },
            "batch": {
                "batch_id": "BATCH-REV-C-QUAL-001",
                "batch_status": "qualification_monitoring",
                "qualification_batch": True,
            },
            "defect_history": {
                "defect_code": "none",
                "same_defect_count_30d": 0,
                "same_defect_count_90d": 0,
            },
            "repair_records": {},
            "maintenance_logs": {},
            "final_dispositions": {},
            "computer_vision": {
                "defect_detected": False,
                "number_of_detection_boxes": 0,
                "estimated_defect_regions": 0,
                "number_of_defects": 0,
                "highest_confidence": 0.0,
                "highest_severity_hint": "none",
                "overall_status_hint": "no_defect_detected",
                "primary_defect_code": "",
                "primary_confidence": 0.0,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
    "TC-006": {
        "description": "SH linked to supplier lot across lines/factories — should recommend supplier_containment",
        "expected_action": "supplier_containment",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "post-etching PCB inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "automotive control PCB with dense conductor routing",
                "affected_system": "body_control",
                "safety_criticality": "high",
                "revision": "Rev-B",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_name": "Demo PCB Supplier",
                "supplied_material": "copper-clad laminate",
                "supplier_lot": "LOT-CU-2026-0518-77",
                "supplier_quality_status": "under_review",
                "supplier_quality_case_id": "SQC-2026-0518-042",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
                "batch_status": "in_progress",
                "quarantine_status": "pending",
            },
            "defect_history": {
                "defect_code": "SH",
                "same_defect_count_30d": 18,
                "same_defect_count_90d": 18,
                "affected_lines": "LINE-A3,LINE-B1",
                "affected_factories": "FACTORY-BERLIN-01,FACTORY-DRESDEN-02",
            },
            "repair_records": {},
            "maintenance_logs": {},
            "final_dispositions": {
                "final_disposition_last_cases": "7 rejected; 3 rework_failed",
                "current_case_disposition": "pending_quality_decision",
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 9,
                "estimated_defect_regions": 4,
                "number_of_defects": 4,
                "highest_confidence": 0.8913,
                "highest_severity_hint": "high",
                "overall_status_hint": "critical_defect_detected",
                "primary_defect_code": "SH",
                "primary_confidence": 0.8913,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
    "TC-007": {
        "description": "CFO night shift pattern at AOI-04 — should recommend clean_station",
        "expected_action": "clean_station",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "post-etching PCB inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "machine_id": "AOI-MACHINE-04",
                "station_function": "automated optical inspection after etching",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "PCB with visible conductor regions after etching, high contamination sensitivity",
                "affected_system": "body_control",
                "revision": "Rev-B",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_quality_status": "normal",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
            },
            "defect_history": {
                "defect_code": "CFO",
                "same_defect_count_30d": 6,
                "same_defect_count_90d": 6,
                "most_common_shift": "Night",
                "scope": "same_station_same_product",
            },
            "repair_records": {},
            "maintenance_logs": {
                "latest_maintenance_id": "MNT-2026-00516",
                "technician_note": "dust accumulation near board loading area",
                "full_cleaning_status": "not_done",
            },
            "final_dispositions": {
                "current_case_disposition": "manual_review_pending",
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 5,
                "estimated_defect_regions": 3,
                "number_of_defects": 3,
                "highest_confidence": 0.8547,
                "highest_severity_hint": "medium_high",
                "overall_status_hint": "manual_review_recommended",
                "primary_defect_code": "CFO",
                "primary_confidence": 0.8547,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
    "TC-008": {
        "description": "SH + OP multi-critical, batch family pattern — should reject",
        "expected_action": "reject",
        "alt_actions": ["escalate_engineering"],
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
                "factory_type": "automotive electronics manufacturing",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "post-etching PCB inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "automotive body control PCB",
                "affected_system": "body_control",
                "safety_criticality": "high",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_quality_status": "under_review",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
                "batch_status": "in_progress",
            },
            "defect_history": {
                "defect_code": "SH",
                "same_defect_count_30d": 3,
                "same_defect_count_90d": 3,
                "multi_critical_defect_count_14d": 3,
                "scope": "same_batch_family_same_product",
            },
            "repair_records": {
                "engineering_rework_failed": 1,
            },
            "maintenance_logs": {},
            "final_dispositions": {
                "final_disposition_last_cases": "2 rejected; 1 engineering_rework_failed",
                "current_case_disposition": "pending_quality_decision",
            },
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 6,
                "estimated_defect_regions": 3,
                "number_of_defects": 3,
                "highest_confidence": 0.7234,
                "highest_severity_hint": "high",
                "overall_status_hint": "critical_defect_detected",
                "primary_defect_code": "SH",
                "primary_confidence": 0.7234,
                "secondary_defect_code": "OP",
                "secondary_confidence": 0.6812,
                "detections": [],
            },
        },
    },
    "TC-009": {
        "description": "MB low confidence (33.75%), no history — should recommend manual_review",
        "expected_action": "manual_review",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
            },
            "production_line": {
                "production_line_id": "LINE-C2",
                "line_function": "secondary PCB inspection line",
            },
            "station": {
                "station_id": "AOI-09",
                "station_type": "automated_optical_inspection",
            },
            "product": {
                "product_model": "PCB-X200",
                "part_number": "PN-PCB-X200-REV-A",
                "product_name": "Lighting Control PCB",
                "product_description": "PCB used in the vehicle lighting control module",
                "affected_system": "lighting_control",
                "safety_criticality": "medium",
                "revision": "Rev-A",
            },
            "supplier": {
                "supplier_id": "SUP-055",
                "supplier_quality_status": "normal",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-19-C",
                "batch_status": "in_progress",
            },
            "defect_history": {
                "defect_code": "MB",
                "same_defect_count_30d": 0,
                "same_defect_count_90d": 0,
                "scope": "same_product_same_line_same_station",
            },
            "repair_records": {},
            "maintenance_logs": {},
            "final_dispositions": {},
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 1,
                "estimated_defect_regions": 1,
                "number_of_defects": 1,
                "highest_confidence": 0.3375,
                "highest_severity_hint": "low",
                "overall_status_hint": "low_confidence_detection",
                "primary_defect_code": "MB",
                "primary_confidence": 0.3375,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
    "TC-010": {
        "description": "HB in test coupon area — should recommend log_and_monitor",
        "expected_action": "log_and_monitor",
        "payload": {
            "factory": {
                "factory_id": "FACTORY-BERLIN-01",
                "factory_name": "Berlin Electronics Plant",
            },
            "production_line": {
                "production_line_id": "LINE-A3",
                "line_function": "post-etching PCB inspection",
            },
            "station": {
                "station_id": "AOI-04",
                "station_type": "automated_optical_inspection",
            },
            "product": {
                "product_model": "PCB-X100",
                "part_number": "PN-PCB-X100-REV-B",
                "product_name": "Main Control PCB",
                "product_description": "automotive body control PCB with separate functional circuit area and process-monitoring test coupon area",
                "board_zone_map_available": True,
                "critical_features": "functional circuit area separated from test coupon area",
                "affected_system": "body_control",
            },
            "supplier": {
                "supplier_id": "SUP-042",
                "supplier_quality_status": "normal",
            },
            "batch": {
                "batch_id": "BATCH-2026-05-18-A",
                "batch_status": "in_progress",
            },
            "defect_history": {
                "defect_code": "HB",
                "same_defect_count_30d": 4,
                "same_defect_count_90d": 4,
                "board_zone": "test_coupon_area",
                "functional_zone": False,
            },
            "repair_records": {},
            "maintenance_logs": {},
            "final_dispositions": {},
            "computer_vision": {
                "defect_detected": True,
                "number_of_detection_boxes": 3,
                "estimated_defect_regions": 3,
                "number_of_defects": 3,
                "highest_confidence": 0.8756,
                "highest_severity_hint": "medium",
                "overall_status_hint": "defect_detected",
                "primary_defect_code": "HB",
                "primary_confidence": 0.8756,
                "secondary_defect_code": "",
                "secondary_confidence": 0.0,
                "detections": [],
            },
        },
    },
}


def run_test(tc_id: str, tc: dict, dry_run: bool = False) -> dict:
    """Run a single test case against the API. Returns result dict."""
    expected = tc["expected_action"]
    alt_actions = tc.get("alt_actions", [])
    acceptable = [expected] + alt_actions

    print(f"\n{'='*70}")
    print(f"  {tc_id}: {tc['description']}")
    print(f"  Expected: {expected}" + (f" (also accept: {', '.join(alt_actions)})" if alt_actions else ""))
    print(f"{'='*70}")

    if dry_run:
        print(json.dumps(tc["payload"], indent=2)[:500] + "\n  ...")
        return {"tc_id": tc_id, "status": "dry_run"}

    start = time.time()
    try:
        resp = httpx.post(API_URL, json=tc["payload"], timeout=180.0)
        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}: {resp.text[:300]}")
            return {"tc_id": tc_id, "status": "error", "http_status": resp.status_code}

        result = resp.json()
        action = result.get("recommended_action", "MISSING")

        passed = action in acceptable
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[-]"

        print(f"  {symbol} Action: {action} ({status}) | {elapsed:.1f}s")
        print(f"  Summary: {result.get('detected_defect_summary', '')[:200]}")
        print(f"  Justification: {result.get('justification', '')[:200]}")
        print(f"  SOPs: {result.get('sop_references', [])}")
        print(f"  Confidence: {result.get('confidence_assessment', {})}")

        return {
            "tc_id": tc_id,
            "status": status,
            "expected": expected,
            "actual": action,
            "elapsed": elapsed,
            "result": result,
        }

    except httpx.ConnectError:
        print("  ERROR: Cannot connect to API. Is the server running?")
        print("  Start with: uvicorn main:app --host 0.0.0.0 --port 8000")
        return {"tc_id": tc_id, "status": "connection_error"}
    except httpx.ReadTimeout:
        elapsed = time.time() - start
        print(f"  ERROR: Timeout after {elapsed:.1f}s")
        return {"tc_id": tc_id, "status": "timeout"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"tc_id": tc_id, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run PCB defect test cases against the API")
    parser.add_argument("--test", type=str, help="Run a specific test case (e.g., TC-001)")
    parser.add_argument("--dry-run", action="store_true", help="Print payloads without calling API")
    args = parser.parse_args()

    if args.test:
        if args.test not in TEST_CASES:
            print(f"Unknown test case: {args.test}. Available: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)
        cases = {args.test: TEST_CASES[args.test]}
    else:
        cases = TEST_CASES

    results = []
    for tc_id, tc in cases.items():
        results.append(run_test(tc_id, tc, dry_run=args.dry_run))

    if not args.dry_run:
        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = sum(1 for r in results if r["status"] == "FAIL")
        errors = sum(1 for r in results if r["status"] not in ("PASS", "FAIL", "dry_run"))
        total = len(results)

        for r in results:
            if r["status"] in ("PASS", "FAIL"):
                symbol = "[+]" if r["status"] == "PASS" else "[-]"
                print(f"  {symbol} {r['tc_id']}: expected={r['expected']}, got={r['actual']} ({r['elapsed']:.1f}s)")
            else:
                print(f"  [!] {r['tc_id']}: {r['status']}")

        print(f"\n  Result: {passed}/{total} passed, {failed} failed, {errors} errors")


if __name__ == "__main__":
    main()
