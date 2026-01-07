"""
PDF Report Generator for MF Classification Results.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from pathlib import Path
from datetime import datetime
from typing import Dict
import config


def generate_report(results: Dict, output_path: Path) -> Path:
    """
    Generate a PDF report for the classification results.
    
    Args:
        results: Dictionary containing prediction results
        output_path: Path to save the PDF file
        
    Returns:
        Path to the generated PDF file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a5276')
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2c3e50')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8
    )
    
    # Build document content
    story = []
    
    # Title
    story.append(Paragraph(config.PDF_TITLE, title_style))
    story.append(Paragraph(config.PDF_INSTITUTION, subtitle_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    
    patient_data = [
        ["Patient ID:", results['patient_name']],
        ["Analysis Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Classification Result
    story.append(Paragraph("Classification Result", heading_style))
    
    # Result box with color based on prediction
    prediction = results['predicted_class']
    confidence = results['confidence'] * 100
    
    if prediction == "MF":
        result_color = colors.HexColor('#e74c3c')  # Red for MF
        result_text = f"<b>MYCOSIS FUNGOIDES (MF)</b>"
    else:
        result_color = colors.HexColor('#27ae60')  # Green for Non-MF
        result_text = f"<b>NON-MYCOSIS FUNGOIDES</b>"
    
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Normal'],
        fontSize=18,
        alignment=TA_CENTER,
        textColor=colors.white,
        backColor=result_color,
        borderPadding=15
    )
    
    result_data = [[Paragraph(result_text, result_style)]]
    result_table = Table(result_data, colWidths=[5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), result_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 20),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 15))
    
    # Confidence
    confidence_text = f"Confidence: <b>{confidence:.1f}%</b>"
    story.append(Paragraph(confidence_text, ParagraphStyle(
        'Confidence',
        parent=styles['Normal'],
        fontSize=14,
        alignment=TA_CENTER
    )))
    story.append(Spacer(1, 20))
    
    # Probability Details
    story.append(Paragraph("Probability Analysis", heading_style))
    
    mf_prob = results['mf_probability'] * 100
    nonmf_prob = results['nonmf_probability'] * 100
    
    prob_data = [
        ["Class", "Probability"],
        ["Mycosis Fungoides (MF)", f"{mf_prob:.1f}%"],
        ["Non-MF", f"{nonmf_prob:.1f}%"],
    ]
    
    prob_table = Table(prob_data, colWidths=[3*inch, 2*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    # Model Details
    story.append(Paragraph("Analysis Details", heading_style))
    
    x10_mf_prob = results['x10_probability'] * 100
    x20_mf_prob = results['x20_probability'] * 100
    fusion_weight = results['fusion_weight']
    
    details_data = [
        ["Parameter", "Value"],
        ["x10 Model MF Probability", f"{x10_mf_prob:.1f}%"],
        ["x20 Model MF Probability", f"{x20_mf_prob:.1f}%"],
        ["x10 Images Analyzed", str(results['n_x10_images'])],
        ["x20 Images Analyzed", str(results['n_x20_images'])],
        ["x10 Patches Extracted", str(results['n_x10_patches'])],
        ["x20 Patches Extracted", str(results['n_x20_patches'])],
        ["Fusion Weight (x10 : x20)", f"{fusion_weight:.2f} : {1-fusion_weight:.2f}"],
    ]
    
    details_table = Table(details_data, colWidths=[3*inch, 2*inch])
    details_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceBefore=20
    )
    
    disclaimer_text = """
    <b>DISCLAIMER:</b> This report is generated by an AI-assisted diagnostic tool and is intended 
    to support clinical decision-making only. It should not be used as the sole basis for diagnosis. 
    Final diagnosis must be made by a qualified pathologist considering all clinical and pathological findings.
    """
    story.append(Paragraph(disclaimer_text, disclaimer_style))
    
    # Build PDF
    doc.build(story)
    
    return output_path
