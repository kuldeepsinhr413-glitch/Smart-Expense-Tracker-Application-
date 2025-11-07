# Smart-Expense-Tracker-Application-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ExpenseTracker:
    """
    A comprehensive expense tracking system that helps users log, analyze,
    and visualize their spending patterns.
    """
    
    def _init_(self, csv_file='expenses.csv'):
        """Initialize the ExpenseTracker with a CSV file."""
        self.csv_file = csv_file
        self.expenses_data = []
        self.valid_categories = ['Food', 'Transport', 'Utilities', 'Entertainment', 'Shopping', 'Healthcare', 'Other']
        
        # Load existing data if file exists
        if os.path.exists(self.csv_file):
            self.load_data()
    
    def add_expense(self, date, amount, category, description):
        """
        Logs new expenses into the dataset with validation.
        
        Parameters:
        - date (str): Date in YYYY-MM-DD format
        - amount (float): Expense amount
        - category (str): Expense category
        - description (str): Brief description of expense
        """
        # Validate date format
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            print("❌ Error: Date must be in YYYY-MM-DD format!")
            return False
        
        # Validate amount (must be positive)
        if not isinstance(amount, (int, float)) or amount <= 0:
            print("❌ Error: Amount must be a positive number!")
            return False
        
        # Validate category
        if category not in self.valid_categories:
            print(f"❌ Error: Category must be one of {self.valid_categories}")
            return False
        
        # Validate description
        if not description or description.strip() == "":
            print("❌ Error: Description cannot be empty!")
            return False
        
        # Add expense to dataset
        expense = {
            'Date': date,
            'Amount': float(amount),
            'Category': category,
            'Description': description.strip()
        }
        self.expenses_data.append(expense)
        print(f"✅ Expense added successfully: ${amount:.2f} for {category}")
        return True
    
    def save_to_csv(self):
        """Save expenses to CSV file."""
        if not self.expenses_data:
            print("⚠ No expenses to save!")
            return
        
        df = pd.DataFrame(self.expenses_data)
        df.to_csv(self.csv_file, index=False)
        print(f"💾 Data saved to {self.csv_file}")
    
    def load_data(self):
        """Load existing data from expenses.csv into a Pandas DataFrame."""
        try:
            df = pd.read_csv(self.csv_file)
            
            # Handle missing or invalid data entries
            df = df.dropna(subset=['Date', 'Amount', 'Category'])
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            df = df[df['Amount'] > 0]
            
            self.expenses_data = df.to_dict('records')
            print(f"📂 Loaded {len(self.expenses_data)} expenses from {self.csv_file}")
        except Exception as e:
            print(f"⚠ Error loading data: {e}")
            self.expenses_data = []
    
    def get_summary(self):
        """
        Provides a summary of total and average expenses.
        
        Returns:
        - Dictionary with total, average, and category-wise spending
        """
        if not self.expenses_data:
            print("⚠ No expenses recorded yet!")
            return None
        
        df = pd.DataFrame(self.expenses_data)
        
        # Calculate total and average using NumPy
        amounts = np.array(df['Amount'])
        total_expenses = np.sum(amounts)
        average_expense = np.mean(amounts)
        
        # Category-wise analysis using Pandas
        category_totals = df.groupby('Category')['Amount'].sum().to_dict()
        
        summary = {
            'total_expenses': total_expenses,
            'average_expense': average_expense,
            'num_expenses': len(amounts),
            'category_totals': category_totals
        }
        
        print("\n" + "="*50)
        print("📊 EXPENSE SUMMARY")
        print("="*50)
        print(f"Total Expenses: ${total_expenses:.2f}")
        print(f"Average Expense: ${average_expense:.2f}")
        print(f"Number of Transactions: {len(amounts)}")
        print("\nCategory-wise Spending:")
        for cat, total in category_totals.items():
            percentage = (total / total_expenses) * 100
            print(f"  • {cat}: ${total:.2f} ({percentage:.1f}%)")
        print("="*50 + "\n")
        
        return summary
    
    def filter_expenses(self, condition):
        """
        Filters expenses by category, date range, or amount.
        
        Parameters:
        - condition (dict): Dictionary with filter criteria
          Examples: {'category': 'Food'}, {'min_amount': 50}, 
                   {'start_date': '2024-01-01', 'end_date': '2024-12-31'}
        """
        if not self.expenses_data:
            print("⚠ No expenses to filter!")
            return []
        
        df = pd.DataFrame(self.expenses_data)
        df['Date'] = pd.to_datetime(df['Date'])
        filtered_df = df.copy()
        
        # Filter by category
        if 'category' in condition:
            filtered_df = filtered_df[filtered_df['Category'] == condition['category']]
        
        # Filter by minimum amount
        if 'min_amount' in condition:
            filtered_df = filtered_df[filtered_df['Amount'] >= condition['min_amount']]
        
        # Filter by maximum amount
        if 'max_amount' in condition:
            filtered_df = filtered_df[filtered_df['Amount'] <= condition['max_amount']]
        
        # Filter by date range
        if 'start_date' in condition:
            start = pd.to_datetime(condition['start_date'])
            filtered_df = filtered_df[filtered_df['Date'] >= start]
        
        if 'end_date' in condition:
            end = pd.to_datetime(condition['end_date'])
            filtered_df = filtered_df[filtered_df['Date'] <= end]
        
        print(f"🔍 Found {len(filtered_df)} matching expenses")
        return filtered_df.to_dict('records')
    
    def generate_report(self):
        """Outputs a summary report with key metrics."""
        if not self.expenses_data:
            print("⚠ No data available for report generation!")
            return
        
        df = pd.DataFrame(self.expenses_data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print("\n" + "="*60)
        print("📈 COMPREHENSIVE EXPENSE REPORT")
        print("="*60)
        
        # Top spending categories
        top_categories = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        print("\n🏆 Top Spending Categories:")
        for i, (cat, amount) in enumerate(top_categories.items(), 1):
            print(f"  {i}. {cat}: ${amount:.2f}")
        
        # Monthly analysis
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_spending = df.groupby('Month')['Amount'].sum()
        print("\n📅 Monthly Spending:")
        for month, amount in monthly_spending.items():
            print(f"  • {month}: ${amount:.2f}")
        
        # Spending statistics using NumPy
        amounts = np.array(df['Amount'])
        print(f"\n📊 Statistical Analysis:")
        print(f"  • Median Expense: ${np.median(amounts):.2f}")
        print(f"  • Highest Expense: ${np.max(amounts):.2f}")
        print(f"  • Lowest Expense: ${np.min(amounts):.2f}")
        print(f"  • Standard Deviation: ${np.std(amounts):.2f}")
        
        print("="*60 + "\n")
    
    def visualize_data(self):
        """
        Generate comprehensive visualizations using Matplotlib and Seaborn.
        """
        if not self.expenses_data:
            print("⚠ No data available for visualization!")
            return
        
        df = pd.DataFrame(self.expenses_data)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(16, 10))
        
        # 1. Bar Chart: Total expenses by category
        plt.subplot(2, 2, 1)
        category_totals = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        sns.barplot(x=category_totals.values, y=category_totals.index, palette='viridis')
        plt.title('Total Expenses by Category', fontsize=14, fontweight='bold')
        plt.xlabel('Amount ($)')
        plt.ylabel('Category')
        
        # 2. Line Graph: Spending trends over time
        plt.subplot(2, 2, 2)
        daily_spending = df.groupby('Date')['Amount'].sum().sort_index()
        plt.plot(daily_spending.index, daily_spending.values, marker='o', linewidth=2, markersize=4)
        plt.title('Spending Trends Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Pie Chart: Proportional spending distribution
        plt.subplot(2, 2, 3)
        colors = sns.color_palette('pastel')[0:len(category_totals)]
        plt.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%', 
                startangle=90, colors=colors)
        plt.title('Proportional Spending Distribution', fontsize=14, fontweight='bold')
        
        # 4. Histogram: Frequency of expense amounts
        plt.subplot(2, 2, 4)
        plt.hist(df['Amount'], bins=15, edgecolor='black', color='skyblue', alpha=0.7)
        plt.title('Frequency of Expense Amounts', fontsize=14, fontweight='bold')
        plt.xlabel('Expense Amount ($)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('expense_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualizations saved as 'expense_analysis.png'")
        plt.show()


def main():
    """Main function to demonstrate the Expense Tracker functionality."""
    print("\n" + "="*60)
    print("💰 SMART EXPENSE TRACKER APPLICATION")
    print("="*60 + "\n")
    
    # Initialize tracker
    tracker = ExpenseTracker('expenses.csv')
    
    # Interactive menu
    while True:
        print("\n📋 MENU:")
        print("1. Add New Expense")
        print("2. View Summary")
        print("3. Filter Expenses")
        print("4. Generate Report")
        print("5. Visualize Data")
        print("6. Save & Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\n➕ ADD NEW EXPENSE")
            date = input("Date (YYYY-MM-DD): ").strip()
            try:
                amount = float(input("Amount ($): ").strip())
            except ValueError:
                print("❌ Invalid amount!")
                continue
            
            print(f"Categories: {', '.join(tracker.valid_categories)}")
            category = input("Category: ").strip()
            description = input("Description: ").strip()
            
            tracker.add_expense(date, amount, category, description)
        
        elif choice == '2':
            tracker.get_summary()
        
        elif choice == '3':
            print("\n🔍 FILTER OPTIONS:")
            print("1. By Category")
            print("2. By Amount Range")
            print("3. By Date Range")
            
            filter_choice = input("Choose filter (1-3): ").strip()
            condition = {}
            
            if filter_choice == '1':
                category = input(f"Category ({', '.join(tracker.valid_categories)}): ").strip()
                condition['category'] = category
            elif filter_choice == '2':
                min_amt = float(input("Minimum amount: ").strip())
                max_amt = float(input("Maximum amount: ").strip())
                condition['min_amount'] = min_amt
                condition['max_amount'] = max_amt
            elif filter_choice == '3':
                start = input("Start date (YYYY-MM-DD): ").strip()
                end = input("End date (YYYY-MM-DD): ").strip()
                condition['start_date'] = start
                condition['end_date'] = end
            
            results = tracker.filter_expenses(condition)
            for exp in results[:10]:  # Show first 10 results
                print(f"  • {exp['Date']}: ${exp['Amount']:.2f} - {exp['Category']} - {exp['Description']}")
        
        elif choice == '4':
            tracker.generate_report()
        
        elif choice == '5':
            tracker.visualize_data()
        
        elif choice == '6':
            tracker.save_to_csv()
            print("\n👋 Thank you for using Smart Expense Tracker!")
            break
        
        else:
            print("❌ Invalid choice! Please select 1-6.")


if _name_ == "_main_":
    main()
