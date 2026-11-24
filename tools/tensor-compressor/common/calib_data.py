"""Built-in 50-sample enterprise structured-extraction dataset and JSONL loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_sample(
    sample_id: str,
    domain: str,
    document: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    fields = list(expected.keys())
    fields_str = ", ".join(f'"{k}"' for k in fields)
    prompt = (
        f"Extract the following fields from the document as a valid JSON object "
        f"with keys [{fields_str}]. Return ONLY JSON.\n\n"
        f"Document:\n{document}\n\nJSON:\n"
    )
    completion = json.dumps(expected, ensure_ascii=False)
    return {
        "id": sample_id,
        "domain": domain,
        "document": document,
        "fields": fields,
        "expected": expected,
        "prompt": prompt,
        "text": prompt + completion,
    }


_RAW_SAMPLES: list[tuple[str, str, str, dict[str, Any]]] = [
    # 1-10: Enterprise Invoices
    (
        "inv-001",
        "invoice",
        "INVOICE #INV-2026-4410 | Vendor: Northstar Cloud Systems Inc.\n"
        "Billing Period: Oct 1 - Oct 31, 2026 | Due Date: 2026-11-30\n"
        "Line Items: Dedicated A100 Cluster (320 hrs) - $14,850.00\n"
        "Total Amount Due: $14,850.00 USD | Cost Center: CC-402",
        {"vendor": "Northstar Cloud Systems Inc.", "invoice_id": "INV-2026-4410", "amount_usd": "14850.00", "due_date": "2026-11-30"},
    ),
    (
        "inv-002",
        "invoice",
        "BILLING STATEMENT | Supplier: Apex Precision Logistics LLC\n"
        "Ref: APL-99812 | Payment Terms: Net 30 | Due: 2026-12-15\n"
        "Freight & Customs Brokerage Total: $8,420.50 USD",
        {"vendor": "Apex Precision Logistics LLC", "invoice_id": "APL-99812", "amount_usd": "8420.50", "due_date": "2026-12-15"},
    ),
    (
        "inv-003",
        "invoice",
        "Vanguard Security Audits Corp - Statement INV-7741\n"
        "SOC2 Type II Penetration Test & Report | Due Date: 2026-12-01\n"
        "Balance Payable: $22,000.00 USD",
        {"vendor": "Vanguard Security Audits Corp", "invoice_id": "INV-7741", "amount_usd": "22000.00", "due_date": "2026-12-01"},
    ),
    (
        "inv-004",
        "invoice",
        "Invoice No: MKL-3092 | Issued by: Merkle Data Foundry Ltd.\n"
        "ETL Pipeline Consulting (Q4) | Amount Due: $6,750.00 USD\n"
        "Remit by: 2026-11-28",
        {"vendor": "Merkle Data Foundry Ltd.", "invoice_id": "MKL-3092", "amount_usd": "6750.00", "due_date": "2026-11-28"},
    ),
    (
        "inv-005",
        "invoice",
        "Helios Semiconductor Supply Co. | Invoice: HSS-55109\n"
        "HBM3 Thermal Interface Modules (500 units) | Total: $31,200.00 USD\n"
        "Due On: 2026-12-20",
        {"vendor": "Helios Semiconductor Supply Co.", "invoice_id": "HSS-55109", "amount_usd": "31200.00", "due_date": "2026-12-20"},
    ),
    (
        "inv-006",
        "invoice",
        "From: Redline Legal Advisory LLP | Doc ID: RLA-2026-88\n"
        "IP Patent Filing & Review | Net Balance: $11,340.00 USD\n"
        "Payment Deadline: 2026-12-05",
        {"vendor": "Redline Legal Advisory LLP", "invoice_id": "RLA-2026-88", "amount_usd": "11340.00", "due_date": "2026-12-05"},
    ),
    (
        "inv-007",
        "invoice",
        "Cobalt Robotics Maintenance | Invoice #CRM-1182\n"
        "Annual Actuator Calibration Contract | Total Due: $19,500.00 USD\n"
        "Due Date: 2026-12-10",
        {"vendor": "Cobalt Robotics Maintenance", "invoice_id": "CRM-1182", "amount_usd": "19500.00", "due_date": "2026-12-10"},
    ),
    (
        "inv-008",
        "invoice",
        "Vendor: Stratos Telemetry Networks | Invoice Ref: STN-6401\n"
        "Fiber Backhaul Lease (100Gbps) | Grand Total: $4,980.00 USD\n"
        "Payable By: 2026-11-29",
        {"vendor": "Stratos Telemetry Networks", "invoice_id": "STN-6401", "amount_usd": "4980.00", "due_date": "2026-11-29"},
    ),
    (
        "inv-009",
        "invoice",
        "Boreal Cold-Chain Packaging Inc. | Bill #BCP-4409\n"
        "Cryo Insulated Containers | Amount: $9,115.75 USD\n"
        "Due: 2026-12-18",
        {"vendor": "Boreal Cold-Chain Packaging Inc.", "invoice_id": "BCP-4409", "amount_usd": "9115.75", "due_date": "2026-12-18"},
    ),
    (
        "inv-010",
        "invoice",
        "Kestrel Vector Search Labs | Invoice ID: KVS-9014\n"
        "Enterprise Dedicated Index Tier | Total: $3,600.00 USD\n"
        "Due Date: 2026-12-03",
        {"vendor": "Kestrel Vector Search Labs", "invoice_id": "KVS-9014", "amount_usd": "3600.00", "due_date": "2026-12-03"},
    ),
    # 11-20: Commercial Contracts & MSAs
    (
        "ctr-011",
        "contract",
        "MASTER SERVICES AGREEMENT MSA-2026-104\n"
        "Counterparty: Obsidian Analytics Group Inc. | Governing Law: Delaware\n"
        "Effective Date: 2026-06-01 | Term: 24 months | Liability Cap: $500,000",
        {"counterparty": "Obsidian Analytics Group Inc.", "contract_id": "MSA-2026-104", "governing_law": "Delaware", "term_months": "24"},
    ),
    (
        "ctr-012",
        "contract",
        "ENTERPRISE LICENSE AGREEMENT ELA-8820\n"
        "Licensor: Horizon BioCompute Ltd. | Jurisdiction: New York\n"
        "Duration: 36 months | Auto-Renewal: 60 days prior notice",
        {"counterparty": "Horizon BioCompute Ltd.", "contract_id": "ELA-8820", "governing_law": "New York", "term_months": "36"},
    ),
    (
        "ctr-013",
        "contract",
        "DATA PROCESSING ADDENDUM DPA-419\n"
        "Vendor: Clearpath Health Informatics | Law: California\n"
        "Initial Term: 12 months | Breach SLA: 24 hours",
        {"counterparty": "Clearpath Health Informatics", "contract_id": "DPA-419", "governing_law": "California", "term_months": "12"},
    ),
    (
        "ctr-014",
        "contract",
        "STATEMENT OF WORK SOW-2026-77B\n"
        "Partner: Ironclad Supply Chain AI | Governing Jurisdiction: Texas\n"
        "Engagement Length: 18 months | Fixed Fee: $240,000",
        {"counterparty": "Ironclad Supply Chain AI", "contract_id": "SOW-2026-77B", "governing_law": "Texas", "term_months": "18"},
    ),
    (
        "ctr-015",
        "contract",
        "HARDWARE LEASE AGREEMENT HLA-503\n"
        "Lessor: Summit GPU Infrastructure LLC | Venue: Washington\n"
        "Lease Term: 12 months | Monthly Rate: $28,000",
        {"counterparty": "Summit GPU Infrastructure LLC", "contract_id": "HLA-503", "governing_law": "Washington", "term_months": "12"},
    ),
    (
        "ctr-016",
        "contract",
        "MUTUAL NDA MNDA-2026-312\n"
        "Party B: Aether Quantum Photonics Corp. | Governing Law: Massachusetts\n"
        "Confidentiality Term: 60 months",
        {"counterparty": "Aether Quantum Photonics Corp.", "contract_id": "MNDA-2026-312", "governing_law": "Massachusetts", "term_months": "60"},
    ),
    (
        "ctr-017",
        "contract",
        "CHANNEL PARTNER AGREEMENT CPA-901\n"
        "Reseller: Nordic Cloud Distribution AS | Law: England and Wales\n"
        "Contract Term: 24 months | Territory: EMEA",
        {"counterparty": "Nordic Cloud Distribution AS", "contract_id": "CPA-901", "governing_law": "England and Wales", "term_months": "24"},
    ),
    (
        "ctr-018",
        "contract",
        "MANAGED SECURITY AGREEMENT SEC-2026-55\n"
        "Provider: Sentinel ZeroTrust Systems | Jurisdiction: Illinois\n"
        "Term: 36 months | Uptime SLA: 99.99%",
        {"counterparty": "Sentinel ZeroTrust Systems", "contract_id": "SEC-2026-55", "governing_law": "Illinois", "term_months": "36"},
    ),
    (
        "ctr-019",
        "contract",
        "API DISTRIBUTION CONTRACT ADC-618\n"
        "Partner: FinLedger Clearing Corp. | Governing Law: New York\n"
        "Duration: 12 months | Rate Limit: 5000 RPS",
        {"counterparty": "FinLedger Clearing Corp.", "contract_id": "ADC-618", "governing_law": "New York", "term_months": "12"},
    ),
    (
        "ctr-020",
        "contract",
        "CUSTOM MODEL TRAINING AGREEMENT MTA-309\n"
        "Contractor: DeepGrid Energy Modeling Inc. | Law: Colorado\n"
        "Project Term: 6 months | IP Ownership: Client",
        {"counterparty": "DeepGrid Energy Modeling Inc.", "contract_id": "MTA-309", "governing_law": "Colorado", "term_months": "6"},
    ),
    # 21-30: Logistics & Bills of Lading
    (
        "log-021",
        "logistics",
        "BILL OF LADING BOL-88412 | Carrier: Pacific Rim Freight\n"
        "Origin: Shenzhen, CN | Destination: Los Angeles, US\n"
        "Gross Weight: 4250 kg | Container: MSKU-904112",
        {"bol_number": "BOL-88412", "carrier": "Pacific Rim Freight", "origin": "Shenzhen, CN", "destination": "Los Angeles, US", "weight_kg": "4250"},
    ),
    (
        "log-022",
        "logistics",
        "AIR WAYBILL AWB-33019 | Carrier: Atlas Air Cargo\n"
        "From: Frankfurt, DE | To: Chicago, US\n"
        "Chargeable Weight: 890 kg | Priority: Express",
        {"bol_number": "AWB-33019", "carrier": "Atlas Air Cargo", "origin": "Frankfurt, DE", "destination": "Chicago, US", "weight_kg": "890"},
    ),
    (
        "log-023",
        "logistics",
        "FREIGHT MANIFEST BOL-77201 | Carrier: Maersk Intermodal\n"
        "Port of Loading: Rotterdam, NL | Port of Discharge: Newark, US\n"
        "Total Weight: 18400 kg | Hazardous: No",
        {"bol_number": "BOL-77201", "carrier": "Maersk Intermodal", "origin": "Rotterdam, NL", "destination": "Newark, US", "weight_kg": "18400"},
    ),
    (
        "log-024",
        "logistics",
        "WAYBILL BOL-55190 | Carrier: FedEx Trade Networks\n"
        "Origin: Taipei, TW | Destination: Austin, US\n"
        "Net Weight: 1320 kg | Contents: Server Motherboards",
        {"bol_number": "BOL-55190", "carrier": "FedEx Trade Networks", "origin": "Taipei, TW", "destination": "Austin, US", "weight_kg": "1320"},
    ),
    (
        "log-025",
        "logistics",
        "SHIPPING ORDER BOL-61044 | Carrier: DHL Global Forwarding\n"
        "Origin: Tokyo, JP | Destination: Seattle, US\n"
        "Weight: 2150 kg | Incoterms: DDP",
        {"bol_number": "BOL-61044", "carrier": "DHL Global Forwarding", "origin": "Tokyo, JP", "destination": "Seattle, US", "weight_kg": "2150"},
    ),
    (
        "log-026",
        "logistics",
        "RAIL CONSIGNMENT BOL-90421 | Carrier: Union Pacific Logistics\n"
        "Origin: Dallas, US | Destination: Denver, US\n"
        "Gross Weight: 9800 kg | Seal: UP-4410",
        {"bol_number": "BOL-90421", "carrier": "Union Pacific Logistics", "origin": "Dallas, US", "destination": "Denver, US", "weight_kg": "9800"},
    ),
    (
        "log-027",
        "logistics",
        "AIR FREIGHT AWB-11902 | Carrier: Kuehne+Nagel\n"
        "Origin: Munich, DE | Destination: Boston, US\n"
        "Weight: 640 kg | Temp Control: 2-8C",
        {"bol_number": "AWB-11902", "carrier": "Kuehne+Nagel", "origin": "Munich, DE", "destination": "Boston, US", "weight_kg": "640"},
    ),
    (
        "log-028",
        "logistics",
        "SEA WAYBILL BOL-48210 | Carrier: Hapag-Lloyd\n"
        "Origin: Busan, KR | Destination: Oakland, US\n"
        "Cargo Weight: 15200 kg | Pallets: 24",
        {"bol_number": "BOL-48210", "carrier": "Hapag-Lloyd", "origin": "Busan, KR", "destination": "Oakland, US", "weight_kg": "15200"},
    ),
    (
        "log-029",
        "logistics",
        "CUSTOMS MANIFEST BOL-29384 | Carrier: Expeditors International\n"
        "Origin: Singapore, SG | Destination: San Jose, US\n"
        "Total Weight: 3100 kg | HS Code: 8471.50",
        {"bol_number": "BOL-29384", "carrier": "Expeditors International", "origin": "Singapore, SG", "destination": "San Jose, US", "weight_kg": "3100"},
    ),
    (
        "log-030",
        "logistics",
        "TRUCKLOAD BOL-73915 | Carrier: J.B. Hunt Transport\n"
        "Origin: Atlanta, US | Destination: Ashburn, US\n"
        "Weight: 7450 kg | Dock: Bay 14",
        {"bol_number": "BOL-73915", "carrier": "J.B. Hunt Transport", "origin": "Atlanta, US", "destination": "Ashburn, US", "weight_kg": "7450"},
    ),
    # 31-40: Regulatory & Compliance Filings
    (
        "cmp-031",
        "compliance",
        "REGULATORY NOTICE CMP-2026-01 | Framework: SOC2 Type II\n"
        "Entity: Meridian FinTech Corp. | Severity: Low | Status: Remediated\n"
        "Finding: Key rotation interval 95 days vs 90-day policy.",
        {"notice_id": "CMP-2026-01", "framework": "SOC2 Type II", "entity": "Meridian FinTech Corp.", "severity": "Low", "status": "Remediated"},
    ),
    (
        "cmp-032",
        "compliance",
        "AUDIT FINDING CMP-2026-08 | Framework: HIPAA\n"
        "Entity: Beacon Clinical Systems | Severity: High | Status: Open\n"
        "Finding: Unencrypted staging bucket detected in us-east-1.",
        {"notice_id": "CMP-2026-08", "framework": "HIPAA", "entity": "Beacon Clinical Systems", "severity": "High", "status": "Open"},
    ),
    (
        "cmp-033",
        "compliance",
        "COMPLIANCE REPORT CMP-2026-14 | Framework: ISO 27001\n"
        "Entity: VaultGrid Storage Inc. | Severity: Medium | Status: In Progress\n"
        "Finding: MFA enrollment missing for 3 contractor accounts.",
        {"notice_id": "CMP-2026-14", "framework": "ISO 27001", "entity": "VaultGrid Storage Inc.", "severity": "Medium", "status": "In Progress"},
    ),
    (
        "cmp-034",
        "compliance",
        "PRIVACY ASSESSMENT CMP-2026-22 | Framework: GDPR\n"
        "Entity: EuroRetail Analytics GmbH | Severity: Low | Status: Closed\n"
        "Finding: Cookie consent banner updated for DE/FR locales.",
        {"notice_id": "CMP-2026-22", "framework": "GDPR", "entity": "EuroRetail Analytics GmbH", "severity": "Low", "status": "Closed"},
    ),
    (
        "cmp-035",
        "compliance",
        "SECURITY CONTROL CMP-2026-31 | Framework: PCI-DSS 4.0\n"
        "Entity: PayStream Gateway LLC | Severity: Critical | Status: Open\n"
        "Finding: Legacy TLS 1.1 endpoint still reachable on port 8443.",
        {"notice_id": "CMP-2026-31", "framework": "PCI-DSS 4.0", "entity": "PayStream Gateway LLC", "severity": "Critical", "status": "Open"},
    ),
    (
        "cmp-036",
        "compliance",
        "FEDRAMP REVIEW CMP-2026-40 | Framework: FedRAMP Moderate\n"
        "Entity: CivicCloud Defense Inc. | Severity: Medium | Status: Remediated\n"
        "Finding: FIPS 140-3 module enabled on ingress proxy.",
        {"notice_id": "CMP-2026-40", "framework": "FedRAMP Moderate", "entity": "CivicCloud Defense Inc.", "severity": "Medium", "status": "Remediated"},
    ),
    (
        "cmp-037",
        "compliance",
        "AI GOVERNANCE LOG CMP-2026-47 | Framework: EU AI Act\n"
        "Entity: TalentScore HR AI Ltd. | Severity: High | Status: In Progress\n"
        "Finding: Bias evaluation dataset documentation incomplete.",
        {"notice_id": "CMP-2026-47", "framework": "EU AI Act", "entity": "TalentScore HR AI Ltd.", "severity": "High", "status": "In Progress"},
    ),
    (
        "cmp-038",
        "compliance",
        "ATTESTATION CMP-2026-53 | Framework: NIST 800-53\n"
        "Entity: GridPulse SCADA Corp. | Severity: Low | Status: Closed\n"
        "Finding: Quarterly access review signed off by CISO.",
        {"notice_id": "CMP-2026-53", "framework": "NIST 800-53", "entity": "GridPulse SCADA Corp.", "severity": "Low", "status": "Closed"},
    ),
    (
        "cmp-039",
        "compliance",
        "COMPLIANCE ALERT CMP-2026-61 | Framework: SOX 404\n"
        "Entity: LedgerWorks ERP Inc. | Severity: Medium | Status: Open\n"
        "Finding: Segregation of duties conflict in prod deploy pipeline.",
        {"notice_id": "CMP-2026-61", "framework": "SOX 404", "entity": "LedgerWorks ERP Inc.", "severity": "Medium", "status": "Open"},
    ),
    (
        "cmp-040",
        "compliance",
        "PRIVACY AUDIT CMP-2026-69 | Framework: CCPA\n"
        "Entity: AdMetric Insights Corp. | Severity: Low | Status: Remediated\n"
        "Finding: Data deletion webhook latency reduced below 48h.",
        {"notice_id": "CMP-2026-69", "framework": "CCPA", "entity": "AdMetric Insights Corp.", "severity": "Low", "status": "Remediated"},
    ),
    # 41-50: SaaS Renewals & GPU Cloud Subscriptions
    (
        "sas-041",
        "saas_billing",
        "SAAS RENEWAL NOTICE | Product: Snowflake Enterprise\n"
        "Account ID: SNF-9921 | Seats: 120 | Annual Spend: $144,000\n"
        "Renewal Date: 2027-01-15 | Auto-Renew: Yes",
        {"product": "Snowflake Enterprise", "account_id": "SNF-9921", "annual_spend_usd": "144000", "renewal_date": "2027-01-15"},
    ),
    (
        "sas-042",
        "saas_billing",
        "SUBSCRIPTION ORDER | Product: Datadog APM Pro\n"
        "Account ID: DD-4018 | Hosts: 250 | Annual Total: $78,500\n"
        "Renewal Date: 2026-12-31 | Billing: Annual",
        {"product": "Datadog APM Pro", "account_id": "DD-4018", "annual_spend_usd": "78500", "renewal_date": "2026-12-31"},
    ),
    (
        "sas-043",
        "saas_billing",
        "ENTERPRISE RENEWAL | Product: Okta Workforce Identity\n"
        "Account ID: OKT-7730 | Users: 850 | Contract Value: $61,200\n"
        "Renewal Date: 2027-02-01",
        {"product": "Okta Workforce Identity", "account_id": "OKT-7730", "annual_spend_usd": "61200", "renewal_date": "2027-02-01"},
    ),
    (
        "sas-044",
        "saas_billing",
        "CLOUD COMMITMENT | Product: Weights & Biases Enterprise\n"
        "Account ID: WNB-2290 | Seats: 45 | Annual Spend: $54,000\n"
        "Renewal Date: 2027-01-01",
        {"product": "Weights & Biases Enterprise", "account_id": "WNB-2290", "annual_spend_usd": "54000", "renewal_date": "2027-01-01"},
    ),
    (
        "sas-045",
        "saas_billing",
        "RENEWAL QUOTE | Product: Cloudflare Enterprise WAF\n"
        "Account ID: CF-5104 | Domains: 18 | Annual Fee: $36,000\n"
        "Renewal Date: 2026-12-15",
        {"product": "Cloudflare Enterprise WAF", "account_id": "CF-5104", "annual_spend_usd": "36000", "renewal_date": "2026-12-15"},
    ),
    (
        "sas-046",
        "saas_billing",
        "LICENSE NOTICE | Product: GitHub Enterprise Cloud\n"
        "Account ID: GHE-8832 | Seats: 300 | Annual Cost: $75,600\n"
        "Renewal Date: 2027-03-01",
        {"product": "GitHub Enterprise Cloud", "account_id": "GHE-8832", "annual_spend_usd": "75600", "renewal_date": "2027-03-01"},
    ),
    (
        "sas-047",
        "saas_billing",
        "PLATFORM SUBSCRIPTION | Product: Databricks SQL Serverless\n"
        "Account ID: DBX-6412 | Commitment: $210,000\n"
        "Renewal Date: 2027-01-31",
        {"product": "Databricks SQL Serverless", "account_id": "DBX-6412", "annual_spend_usd": "210000", "renewal_date": "2027-01-31"},
    ),
    (
        "sas-048",
        "saas_billing",
        "SUPPORT RENEWAL | Product: HashiCorp Terraform Enterprise\n"
        "Account ID: HCP-3109 | Workspaces: 500 | Annual Price: $92,000\n"
        "Renewal Date: 2026-12-20",
        {"product": "HashiCorp Terraform Enterprise", "account_id": "HCP-3109", "annual_spend_usd": "92000", "renewal_date": "2026-12-20"},
    ),
    (
        "sas-049",
        "saas_billing",
        "ANNUAL ORDER | Product: Pinecone Dedicated Pods\n"
        "Account ID: PCN-1944 | Pods: 16 | Annual Spend: $48,000\n"
        "Renewal Date: 2027-02-15",
        {"product": "Pinecone Dedicated Pods", "account_id": "PCN-1944", "annual_spend_usd": "48000", "renewal_date": "2027-02-15"},
    ),
    (
        "sas-050",
        "saas_billing",
        "SAAS AGREEMENT | Product: PagerDuty Incident Response\n"
        "Account ID: PD-7055 | Responders: 90 | Annual Spend: $32,400\n"
        "Renewal Date: 2027-01-10",
        {"product": "PagerDuty Incident Response", "account_id": "PD-7055", "annual_spend_usd": "32400", "renewal_date": "2027-01-10"},
    ),
]

BUILTIN_EXTRACTION_SAMPLES: list[dict[str, Any]] = [
    _make_sample(sid, domain, doc, exp) for sid, domain, doc, exp in _RAW_SAMPLES
]


def load_calibration_dataset(
    path: str | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Load a JSONL vertical dataset or return the built-in 50-sample extraction set."""
    if path is None:
        samples = list(BUILTIN_EXTRACTION_SAMPLES)
        return samples[:max_samples] if max_samples is not None else samples

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Calibration JSONL file not found: {jsonl_path}")

    samples: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("id", f"custom-{idx:04d}"))
            domain = str(row.get("domain", "custom"))

            if "prompt" in row and "expected" in row and isinstance(row["expected"], dict):
                expected = row["expected"]
                prompt = str(row["prompt"])
                text = str(row.get("text") or (prompt + json.dumps(expected, ensure_ascii=False)))
                item = {
                    "id": sid,
                    "domain": domain,
                    "document": str(row.get("document", "")),
                    "fields": list(expected.keys()),
                    "expected": expected,
                    "prompt": prompt,
                    "text": text,
                }
                if "split" in row:
                    item["split"] = row["split"]
                samples.append(item)
            elif "input" in row and "output" in row:
                prompt = str(row["input"])
                output = row["output"]
                expected = output if isinstance(output, dict) else {}
                out_str = json.dumps(output, ensure_ascii=False) if isinstance(output, dict) else str(output)
                samples.append(
                    {
                        "id": sid,
                        "domain": domain,
                        "document": prompt,
                        "fields": list(expected.keys()),
                        "expected": expected,
                        "prompt": prompt,
                        "text": f"{prompt}\n{out_str}",
                    }
                )
            elif "messages" in row and isinstance(row["messages"], list):
                joined = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in row["messages"]
                )
                samples.append(
                    {
                        "id": sid,
                        "domain": domain,
                        "document": "",
                        "fields": [],
                        "expected": {},
                        "prompt": joined,
                        "text": joined,
                    }
                )
            elif "text" in row:
                samples.append(
                    {
                        "id": sid,
                        "domain": domain,
                        "document": "",
                        "fields": [],
                        "expected": {},
                        "prompt": str(row["text"])[:256],
                        "text": str(row["text"]),
                    }
                )
            else:
                raise ValueError(
                    f"Unsupported JSONL row schema at line {idx + 1}: keys={list(row.keys())}"
                )

            if max_samples is not None and len(samples) >= max_samples:
                break

    if not samples:
        raise ValueError(f"No valid calibration samples loaded from {jsonl_path}")
    return samples


def split_calib_eval(
    samples: list[dict[str, Any]],
    eval_fraction: float = 0.2,
    smoke_test: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split dataset into calibration and evaluation subsets."""
    if smoke_test:
        # Balanced subset across domains for fast end-to-end verification
        if len(samples) >= 26:
            calib = samples[:16]
            eval_set = samples[-10:]
            return calib, eval_set
        split_idx = max(1, int(len(samples) * 0.7))
        return samples[:split_idx], samples[split_idx:] or samples[-2:]

    calib_tagged = [s for s in samples if s.get("split") == "calib"]
    eval_tagged = [s for s in samples if s.get("split") == "eval"]
    if calib_tagged and eval_tagged:
        return calib_tagged, eval_tagged

    n_eval = max(2, min(10, int(round(len(samples) * eval_fraction))))
    if len(samples) <= n_eval:
        return samples, samples
    return samples[:-n_eval], samples[-n_eval:]
