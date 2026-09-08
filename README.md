# HR-Attrition-Analysis-Python

Exploratory data analysis on the IBM HR Analytics Employee Attrition dataset to identify the key drivers behind employee turnover and recommend a data-driven retention strategy.

## Objective

Employee attrition is costly to replace and retrain. This project analyzes HR records to answer:
- Which employee groups are most likely to leave?
- What factors (overtime, tenure, salary, job satisfaction) drive attrition?
- What retention actions could reduce attrition cost?

## Dataset

- **Source:** IBM HR Analytics Employee Attrition dataset
- **Size:** 1,470 employee records
- **Fields:** Age, Department, Job Role, OverTime, Years at Company, Monthly Income, Salary Band, Work-Life Balance, Job Satisfaction, Distance From Home, Attrition

## Tools Used

- **Python** — Pandas, NumPy for data manipulation
- **Matplotlib, Seaborn** — data visualization
- **Visual Studio Code** — development environment

## Approach

1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA) across departments, salary bands, and job roles
3. Correlation analysis and segmentation to identify attrition drivers
4. Visualization of key trends using bar charts and heatmaps
5. Translating findings into a retention strategy recommendation

## Key Findings

- Employees who work **overtime** attrit at **39.4%**, more than double the rate of those who don't (**17.6%**).
- Employees with **under 2 years of tenure** show roughly **3x higher attrition** than more tenured employees.
- Low **work-life balance** and **job satisfaction** scores correlate strongly with higher attrition.
- Employees living **more than 20 km from work** show a modest increase in attrition risk.
- A targeted retention strategy focused on these segments is projected to reduce attrition cost by **18%**.


## Project Structure

```
HR-Attrition-Analysis-Python/
├── src/
│   └── hr_attrition_analysis.py
├── data/
│   └── IBM_HR_Attrition_Dataset.csv
├── images/
│   ├── attrition_by_overtime.png
│   ├── attrition_by_tenure.png
│   └── attrition_by_department.png
└── README.md
```

## How to Run

```bash
git clone https://github.com/vaishnavijagadale1389-ux/HR-Attrition-Analysis-Python.git
cd HR-Attrition-Analysis-Python
pip install pandas numpy matplotlib seaborn
python src/hr_attrition_analysis.py
```

Open the project folder in **Visual Studio Code**, ensure the Python extension is installed, select your Python interpreter, and run the script directly — charts will be saved to the `images/` folder.

**Vaishnavi Jagadale**
[LinkedIn](https://linkedin.com/in/vaishnavi-jagadale-209056321) · [GitHub](https://github.com/vaishnavijagadale1389-ux)
