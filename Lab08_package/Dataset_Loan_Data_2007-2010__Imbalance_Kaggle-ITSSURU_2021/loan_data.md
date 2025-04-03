# Loan Data

classify and predict whether or not the borrower paid back their loan in full.



Source: https://www.kaggle.com/datasets/itssuru/loan-data

License: https://opendatacommons.org/licenses/dbcl/1-0/ Database Contents License (DbCL) v1.0 - 

Owner: ItsSuru 

Data Source: LendingClub.com

 

# Dataset Description

This dataset contains loan data from LendingClub.com, a platform connecting borrowers with investors, spanning the years 2007 to 2010. It includes information on over 9,500 loans, detailing loan structure, borrower characteristics, and loan repayment status. The data is derived from publicly available information on LendingClub.com. **Source:** [Kaggle - Loan Data](https://www.kaggle.com/itssuru/loan-data)



## About Dataset

# About the data and what to do…

publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, **as an investor you would want to invest in people who showed a profile of having a high probability of paying you back.**

We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from [here](https://www.lendingclub.com/investing/peer-to-peer).

Here are what the columns represent:

- credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
- purpose: The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").
- int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
- installment: The monthly installments owed by the borrower if the loan is funded.
- log.annual.inc: The natural log of the self-reported annual income of the borrower.
- dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
- fico: The FICO credit score of the borrower.
- days.with.cr.line: The number of days the borrower has had a credit line.
- revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
- revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
- inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
- delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
- pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

## Data Dictionary

| Variable          | Explanation                                                  |
| :---------------- | :----------------------------------------------------------- |
| credit_policy     | 1 if the customer meets the credit underwriting criteria; 0 otherwise. |
| purpose           | The purpose of the loan.                                     |
| int_rate          | The interest rate of the loan (higher rates indicate higher risk). |
| installment       | The monthly installments owed by the borrower if the loan is funded. |
| log_annual_inc    | The natural logarithm of the borrower's self-reported annual income. |
| dti               | The borrower's debt-to-income ratio (debt divided by annual income). |
| fico              | The borrower's FICO credit score.                            |
| days_with_cr_line | The number of days the borrower has had a credit line.       |
| revol_bal         | The borrower's revolving balance (unpaid amount at the end of the credit card billing cycle). |
| revol_util        | The borrower's revolving line utilization rate (credit line used relative to total available credit). |
| inq_last_6mths    | The borrower's number of credit inquiries in the last 6 months. |
| delinq_2yrs       | The number of times the borrower was 30+ days past due on a payment in the past 2 years. |
| pub_rec           | The borrower's number of derogatory public records.          |
| not_fully_paid    | 1 if the loan was not fully paid; 0 otherwise.               |