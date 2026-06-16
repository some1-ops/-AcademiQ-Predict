import pandas as pd
from fpdf import FPDF
import io

class PDFReport(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'AcademiQ: Official Academic Forecast Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_academic_report(df_timeline: pd.DataFrame, final_cgpa: float) -> bytes:
    """
    Generates a formal PDF report using FPDF.
    Returns the PDF bytes.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Title / Header
    pdf.set_font("Arial", size=12)
    
    # Final CGPA
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(41, 128, 185) # Blueish
    pdf.cell(0, 10, f"Final Forecasted CGPA: {final_cgpa:.2f} / 5.00", 0, 1, 'C')
    pdf.ln(10)
    
    # Table Header
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 11)
    
    col_widths = [25, 25, 30, 45, 35, 30]
    headers = ["Level", "Semester", "Course", "Actual/Predicted", "Performance", "Grade Point"]
    
    for width, header in zip(col_widths, headers):
        pdf.cell(width, 10, header, 1, 0, 'C')
    pdf.ln()
    
    # Table Content
    pdf.set_font("Arial", size=10)
    for _, row in df_timeline.iterrows():
        level = str(row.get("Level", ""))
        sem = str(row.get("Semester", ""))
        course = str(row.get("Course_Code", ""))
        phase = str(row.get("Phase", ""))
        perf = str(row.get("Predicted_Performance", ""))
        gp = str(row.get("Predicted_Grade_Point", ""))
        
        pdf.cell(col_widths[0], 10, level, 1, 0, 'C')
        pdf.cell(col_widths[1], 10, sem, 1, 0, 'C')
        pdf.cell(col_widths[2], 10, course, 1, 0, 'C')
        pdf.cell(col_widths[3], 10, phase, 1, 0, 'C')
        pdf.cell(col_widths[4], 10, perf, 1, 0, 'C')
        pdf.cell(col_widths[5], 10, gp, 1, 0, 'C')
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')
