"""
PDF Report Generator for MF Classification Results.

Generates a comprehensive PDF report including:
- Malignant vs Benign classification
- Individual class probabilities for all 5 differential diagnoses
- Detailed analysis metrics
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
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
    
    # Determine Malignant vs Benign classification using the same display logic as the app.
    MALIGNANT_CLASSES = {"B cell Lymphoma", "Mycosis Fungoides"}
    prediction = results['predicted_class']
    if prediction == "MF":
        prediction = "Mycosis Fungoides"
    is_malignant = prediction in MALIGNANT_CLASSES
    diagnosis_label = "Malignant" if is_malignant else "Benign"
    diagnosis_color_hex = "#e74c3c" if is_malignant else "#3498db"
    diagnosis_color = colors.HexColor(diagnosis_color_hex)

    class_probs = results['class_probabilities']
    malignant_prob = sum(v for k, v in class_probs.items() if k in MALIGNANT_CLASSES) * 100
    benign_prob = 100 - malignant_prob
    displayed_confidence = malignant_prob if is_malignant else benign_prob
    diagnosis_text = (
        f"<b>{diagnosis_label}: {prediction}</b><br/>"
        f"Confidence: {displayed_confidence:.1f}%"
    )
    
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Normal'],
        fontSize=18,
        alignment=TA_CENTER,
        leading=24,
        textColor=diagnosis_color,
        borderColor=diagnosis_color,
        borderWidth=1.5,
        borderPadding=15,
        backColor=colors.HexColor('#f8f9fa')
    )
    
    result_data = [[Paragraph(diagnosis_text, result_style)]]
    result_table = Table(result_data, colWidths=[5*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 1.5, diagnosis_color),
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
    confidence_text = (
        f"Classification: <font color='{diagnosis_color_hex}'><b>{diagnosis_label}</b></font> | "
        f"Predicted Class: <b>{prediction}</b> ({displayed_confidence:.1f}% confidence)"
    )
    story.append(Paragraph(confidence_text, ParagraphStyle(
        'Confidence',
        parent=styles['Normal'],
        fontSize=12,
        alignment=TA_CENTER
    )))
    story.append(Spacer(1, 20))
    
    # Binary Classification (Malignant vs Benign)
    story.append(Paragraph("Binary Classification Analysis", heading_style))

    binary_data = [
        ["Classification", "Probability"],
        ["Malignant", f"{malignant_prob:.1f}%"],
        ["Benign", f"{benign_prob:.1f}%"],
    ]
    
    binary_table = Table(binary_data, colWidths=[3*inch, 2*inch])
    binary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0, 2), (-1, 2), colors.HexColor('#3498db')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(binary_table)
    story.append(PageBreak())
    
    # Detailed Classification (5-Class Probabilities)
    story.append(Paragraph("Detailed Classification (5 Differential Diagnoses)", heading_style))
    
    # Build probability data for all 5 classes
    prob_data = [["Diagnosis", "Classification", "Probability"]]
    predicted_row_idx = None
    for class_name in class_probs.keys():
        class_prob = class_probs[class_name] * 100
        classification = "Malignant" if class_name in MALIGNANT_CLASSES else "Benign"
        if class_name == prediction:
            predicted_row_idx = len(prob_data)
        prob_data.append([class_name, classification, f"{class_prob:.1f}%"])
    
    prob_table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]

    for row_idx, row in enumerate(prob_data[1:], start=1):
        row_color = colors.HexColor('#e74c3c') if row[1] == "Malignant" else colors.HexColor('#2ecc71')
        prob_table_style.append(('TEXTCOLOR', (1, row_idx), (-1, row_idx), row_color))

    if predicted_row_idx is not None:
        prob_table_style.extend([
            ('BACKGROUND', (0, predicted_row_idx), (-1, predicted_row_idx), colors.HexColor('#ecf0f1')),
            ('FONTNAME', (0, predicted_row_idx), (-1, predicted_row_idx), 'Helvetica-Bold'),
        ])

    prob_table = Table(prob_data, colWidths=[2.2*inch, 1.3*inch, 1.5*inch])
    prob_table.setStyle(TableStyle(prob_table_style))
    story.append(prob_table)
    story.append(Spacer(1, 20))
    
    # Model Details
    story.append(Paragraph("Analysis Details", heading_style))
    
    # Extract model probabilities and fusion details
    fusion_weight = results['fusion_weight_x10']
    
    details_data = [
        ["Parameter", "Value"],
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
