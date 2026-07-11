# Synthetic loan policy dataset for testing and experimentation.

LOAN_POLICIES = [
    {
        "id": "home_loan_01",
        "category": "Home Loan",
        "title": "Apex Home Loan Interest Rates and Tenure",
        "text": """Apex Bank offers home loans starting at 8.45% per annum for salaried individuals and 8.65% per annum for self-employed professionals. The actual interest rate depends on the applicant's credit score:
- Credit Score 750 or above: 8.45% p.a. (Salaried), 8.65% p.a. (Self-Employed)
- Credit Score 700 to 749: 8.75% p.a. (Salaried), 8.95% p.a. (Self-Employed)
- Credit Score 650 to 699: 9.15% p.a. (Salaried), 9.35% p.a. (Self-Employed)
The maximum loan repayment tenure is up to 30 years (or up to retirement age, whichever is earlier). Principal repayment and interest payments are calculated monthly under a reducing balance method. Additional processing fees of 0.5% of the loan amount (capped at INR 10,000) are charged at the time of processing."""
    },
    {
        "id": "home_loan_02",
        "category": "Home Loan",
        "title": "Apex Home Loan Eligibility and LTV Ratio",
        "text": """To qualify for the Apex Home Loan, applicants must meet the following criteria:
- Age: Minimum 21 years at the time of application, and maximum 65 years or retirement age at loan maturity.
- Minimum Monthly Net Salary: INR 25,000 for salaried employees, or Net Annual Income of INR 3,00,000 for self-employed individuals.
- Loan-to-Value (LTV) Ratio Guidelines:
  - For loans up to INR 30 Lakhs: LTV ratio can be up to 90% of the property value.
  - For loans between INR 30 Lakhs and 75 Lakhs: LTV ratio can be up to 80% of the property value.
  - For loans above INR 75 Lakhs: LTV ratio can be up to 75% of the property value.
Co-applicants are permitted (family members like parents, spouse, or siblings). Adding a female co-applicant may qualify the loan for an interest rate concession of 0.05% per annum."""
    },
    {
        "id": "personal_loan_01",
        "category": "Personal Loan",
        "title": "Vanguard Personal Loan Terms and Charges",
        "text": """Vanguard Personal Loans are unsecured loans ranging from INR 50,000 to INR 20,00,000. Key terms include:
- Interest Rate: Rates are fixed, ranging from 10.99% p.a. to 21.00% p.a., determined based on income level and company categorization.
- Repayment Tenure: Offers flexible tenure options from 12 months to 60 months (5 years).
- Processing Fee: A one-time processing fee of 1.5% to 2.5% of the loan amount is applicable.
- Prepayment and Foreclosure: Foreclosure is allowed only after successful payment of 12 monthly EMIs. A foreclosure fee of 4% of the outstanding principal balance will be charged. Part-payments are not permitted."""
    },
    {
        "id": "personal_loan_02",
        "category": "Personal Loan",
        "title": "Vanguard Personal Loan Documentation Requirements",
        "text": """Applicants for Vanguard Personal Loans must submit the following documents:
- Identity Proof: Aadhaar Card, PAN Card, Passport, or Voter ID.
- Address Proof: Utility bills (not older than 3 months), Rent Agreement, or Passport.
- Income Proof (Salaried): Last 3 months' salary slips, Form 16, and bank statements for the last 6 months showing salary credit.
- Income Proof (Self-Employed): Last 2 years' ITR (Income Tax Returns) with balance sheet and profit & loss statements, bank statements for the last 6 months."""
    },
    {
        "id": "auto_loan_01",
        "category": "Auto Loan",
        "title": "Swift Auto Finance Program Guidelines",
        "text": """The Swift Auto Finance Program provides financing for both new and pre-owned passenger vehicles:
- New Cars: Funding up to 90% of the on-road price. Interest rates start at 9.25% p.a. with tenure up to 7 years.
- Pre-owned Cars: Funding up to 80% of the valuation price of the car (certified by an authorized bank valuer). The vehicle age at loan maturity must not exceed 10 years. Interest rates start at 12.50% p.a. with tenure up to 5 years.
- Hypothecation: The financed vehicle will be hypothecated to Swift Auto Finance (bank) until the loan is fully repaid, and a No Objection Certificate (NOC) is issued."""
    },
    {
        "id": "education_loan_01",
        "category": "Education Loan",
        "title": "EduGrow Education Loan Eligibility and Moratorium",
        "text": """EduGrow Education Loans assist students pursuing higher education in India or abroad:
- Eligible Courses: Undergraduate, Postgraduate, and Professional courses from recognized universities.
- Maximum Loan Amount: Up to INR 15 Lakhs for studies in India, and up to INR 30 Lakhs for studies abroad.
- Moratorium Period: Includes a holiday/moratorium period covering the course duration plus 1 year, or 6 months after securing a job (whichever is earlier). During this period, only simple interest is charged.
- Security Requirements:
  - Loans up to INR 4 Lakhs: No security required. Parents must sign as co-borrowers.
  - Loans between INR 4 Lakhs and 7.5 Lakhs: Third-party guarantee required along with parent co-borrowers.
  - Loans above INR 7.5 Lakhs: Tangible collateral security (e.g., land, building, fixed deposit) of equivalent value is mandatory."""
    }
]
