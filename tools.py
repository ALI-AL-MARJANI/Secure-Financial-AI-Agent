from langchain_core.tools import tool

@tool
def get_account_balance(customer_id: str) -> str:
    """
    Fetches the current account balance for a given customer ID
    always ask the user for their customer ID if it is not provided
    """
    # our mock banking database
    mock_db = {
        "CUST123": 5432.50,
        "CUST456": 120500.00,
        "CUST789": 150.75}

    customer_id = customer_id.upper()
    
    if customer_id in mock_db:
        return f"The current balance for {customer_id} is ${mock_db[customer_id]:,.2f}"
    return f"Error: Customer ID '{customer_id}' not found in the system "

@tool
def simulate_mortgage(principal: float, annual_rate: float, years: int) -> str:
    """
    Simulates a monthly mortgage payment based on the loan principal, annual interest rate, and loan term
    """
    monthly_rate = (annual_rate / 100) / 12
    num_payments = years * 12
    
    if monthly_rate == 0:
        payment = principal / num_payments
    else:
        payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
    return f"The estimated monthly mortgage payment is ${payment:,.2f}."

financial_tools = [get_account_balance, simulate_mortgage]