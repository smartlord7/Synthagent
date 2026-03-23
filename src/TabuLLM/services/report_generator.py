from fpdf import FPDF
import pandas as pd
import io
import datetime

class PDFReportGenerator(FPDF):
    def header(self):
        # Dark Banner Header
        self.set_fill_color(33, 37, 41) # Dark Theme
        self.rect(0, 0, 210, 35, 'F')
        
        self.set_y(12)
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(248, 249, 250)
        self.cell(0, 10, 'TabuLLM Data Augmentation Report', border=0, ln=1, align='C')
        
        # Subtitle or Date
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(173, 181, 189)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 5, f'Generated on: {timestamp}', border=0, ln=1, align='C')
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(108, 117, 125)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title, icon_char=""):
        self.ln(5)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(13, 110, 253) # Bootstrap Primary Blue
        self.cell(0, 10, f"{icon_char} {title}" if icon_char else title, ln=1)
        self.set_draw_color(13, 110, 253)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(4)

def generate_pdf_report(filename, stats_grid_vals, description, params_dict, metrics_dict, df_orig, df_aug):
    """
    Generates a professional PDF report detailing dataset stats, parameters used, metrics, and plots.
    
    Args:
        filename (str): Name of the uploaded file.
        stats_grid_vals (dict): Rows, Cols, Missing, Memory.
        description (str): Structured Cached Summary description from LLM.
        params_dict (dict): tone, consistency, uniqueness.
        metrics_dict (dict): Dictionary calculated from calculate_metrics().
        df_orig (pd.DataFrame): Original Dataframe.
        df_aug (pd.DataFrame): Augmented Dataframe.
        
    Returns:
        bytes: Raw PDF document bytes streams.
    """
    pdf = PDFReportGenerator()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_text_color(33, 37, 41)

    # --- 1. Dataset Overview ---
    pdf.section_title("1. Dataset Overview")
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(40, 7, "Filename:", ln=0)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 7, str(filename), ln=1)
    pdf.ln(2)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(248, 249, 250)
    pdf.set_draw_color(222, 226, 230)
    col_width = 45
    row_height = 12
    pdf.cell(col_width, row_height, f"Rows: {stats_grid_vals.get('Rows', 'N/A')}", border=1, fill=True, align='C')
    pdf.cell(5)
    pdf.cell(col_width, row_height, f"Cols: {stats_grid_vals.get('Cols', 'N/A')}", border=1, fill=True, align='C')
    pdf.ln(row_height + 4)
    pdf.cell(col_width, row_height, f"Missing: {stats_grid_vals.get('Missing', 'N/A')}", border=1, fill=True, align='C')
    pdf.cell(5)
    pdf.cell(col_width, row_height, f"Size: {stats_grid_vals.get('Memory', 'N/A')}", border=1, fill=True, align='C')
    pdf.ln(row_height + 6)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, "LLM Structured Analysis Insights:", ln=1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_fill_color(255, 255, 255)
    desc_clean = str(description).replace("**", "")
    pdf.multi_cell(0, 6, desc_clean if desc_clean else "No analysis description generated.", border=1)
    pdf.ln(6)

    # --- 2. Generation Profile Settings ---
    pdf.section_title("2. Generation Parameters")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(233, 236, 239)
    pdf.cell(80, 8, "Parameter", border=1, fill=True)
    pdf.cell(110, 8, "Value Selected", border=1, fill=True, ln=1)
    pdf.set_font('Helvetica', '', 10)
    for k, v in params_dict.items():
        pdf.cell(80, 8, k.replace("_", " ").title(), border=1)
        pdf.cell(110, 8, str(v), border=1, ln=1)
    pdf.ln(8)

    # --- 3. Quality & Fidelity Scores ---
    pdf.section_title("3. Quality & Fidelity Scores")
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(25, 135, 84)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(110, 8, "Metric Score", border=1, fill=True)
    pdf.cell(80, 8, "Result", border=1, fill=True, ln=1, align='C')
    pdf.set_text_color(33, 37, 41)
    pdf.set_font('Helvetica', '', 10)
    for k, v in metrics_dict.items():
        readable_k = k.replace("_", " ").title()
        if isinstance(v, (int, float)):
            formatted_v = f"{v}%" if any(x in k.lower() for x in ["score", "stability", "fidelity", "risk", "completeness"]) else f"{v:.3f}"
        else: formatted_v = str(v)
        pdf.cell(110, 8, readable_k, border=1)
        pdf.cell(80, 8, str(formatted_v), border=1, ln=1, align='C')

    # --- 4. EDA Visuals Appendices ---
    numeric_cols = df_aug.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) >= 2:
        try:
            import plotly.express as px
            pdf.add_page()
            pdf.section_title("4. Correlation Matrix Matrix Analysis Insights")
            
            corr_aug = df_aug[numeric_cols].corr().fillna(0)
            fig_corr = px.imshow(corr_aug, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix index Setup")
            img_bytes = fig_corr.to_image(format="png", width=700, height=500)
            pdf.image(io.BytesIO(img_bytes), x=15, y=40, w=180)
        except Exception as e:
            pdf.ln(5)
            pdf.cell(0, 10, f"Could not embeds Correlation Map: {e}", ln=1)

    # --- 5. Data Head Previews ---
    pdf.add_page()
    pdf.section_title("5. Original Data Snapshot sample index headers")
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(222, 226, 230)
    sample_sub = df_orig.head(5)
    
    # Render mini headers
    c_w = 190 / len(sample_sub.columns) if sample_sub.columns.any() else 20
    for col in sample_sub.columns[:6]: # Limit to first 6 columns to fit page
        pdf.cell(30, 7, str(col)[:10], border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 8)
    for _, row in sample_sub.iterrows():
        for val in row[:6]:
            pdf.cell(30, 6, str(val)[:15], border=1)
        pdf.ln()

    # Augmented preview
    pdf.ln(10)
    pdf.section_title("6. Augmented Data Snapshot index setup headers")
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(222, 226, 230)
    sample_aug = df_aug.head(5)
    for col in sample_aug.columns[:6]:
        pdf.cell(30, 7, str(col)[:10], border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 8)
    for _, row in sample_aug.iterrows():
        for val in row[:6]:
            pdf.cell(30, 6, str(val)[:15], border=1)
        pdf.ln()

    return pdf.output(dest='S')

