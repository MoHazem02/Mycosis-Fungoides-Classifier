"""
Mycosis Fungoides Classifier - Desktop Application
CustomTkinter GUI for patient histology classification.
"""
import customtkinter as ctk 
from tkinter import filedialog, messagebox
from pathlib import Path
import threading
from typing import Optional
import sys
import os
from PIL import Image, ImageTk
import pandas as pd
import joblib

# Add current directory to path for imports when bundled
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)

from classifier import MFClassifier
from report_generator import generate_report




_BLUE_DARK  = ("#1a5f8a", "#0f3a55")   # section-header background
_BLUE_MID   = ("#2980b9", "#1a5276")   # borders, button accents
_GREEN      = ("#1a8c4e", "#145e36")   # MF-positive result colour
_SLATE      = ("#5d6d7e", "#3d4f5e")   # Non-MF result colour
_DIVIDER    = ("#d5d8dc", "#2c3e50")   # thin separator lines





# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class MFClassifierApp(ctk.CTk):
    """Main application window for MF Classifier."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Mycosis Fungoides Classifier")
        self.geometry("800x950")
        self.minsize(700, 850) 
        
        # Initialize classifier
        self.classifier: Optional[MFClassifier] = None
        self.current_results: Optional[dict] = None
        self.selected_folder: Optional[Path] = None
        
        # Create UI
        self._create_widgets()
        
        # Initialize classifier in background
        self._init_classifier_async()
    
    def _create_widgets(self):
        """Create all UI widgets."""
        
        # Main container (regular frame, not scrollable)
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header frame with logos
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(0, 10))
        
        # Load and display logos
        try:
            # CU logo on the left
            cu_path = Path(__file__).parent / "assets" / "CUFE.png"
            if cu_path.exists():
                cu_image = Image.open(cu_path)
                cu_photo = ctk.CTkImage(light_image=cu_image, dark_image=cu_image, size=(90, 90))
                self.cu_label = ctk.CTkLabel(self.header_frame, image=cu_photo, text="")
                self.cu_label.pack(side="left", padx=10)
            
            # Kasr logo on the right
            kasr_path = Path(__file__).parent / "assets" / "Kasr.png"
            if kasr_path.exists():
                kasr_image = Image.open(kasr_path)
                kasr_photo = ctk.CTkImage(light_image=kasr_image, dark_image=kasr_image, size=(90, 90))
                self.kasr_label = ctk.CTkLabel(self.header_frame, image=kasr_photo, text="")
                self.kasr_label.pack(side="right", padx=10)
        except Exception as e:
            print(f"Could not load logos: {e}")
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Mycosis Fungoides Classifier",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        self.title_label.pack(pady=(0, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="AI-Powered Histopathology Analysis",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.subtitle_label.pack(pady=(0, 20))
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self.main_frame)
        self.tabview.pack(fill="both", expand=True, padx=0, pady=10)
        
        # Add tabs
        self.tab_images = self.tabview.add("Images Classifier")
        self.tab_clinical = self.tabview.add("Clinical Notes")
        
        # Configure tabs
        self._create_images_classifier_tab()
        self._create_clinical_notes_tab()
    
    def _create_images_classifier_tab(self):
        """Create the Images Classifier tab UI."""
        tab = self.tab_images
        
        # Create scrollable container for all content
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        
        # Folder selection frame
        self.folder_frame = ctk.CTkFrame(scroll)
        self.folder_frame.pack(fill="x", padx=20, pady=10)
        
        self.folder_label = ctk.CTkLabel(
            self.folder_frame,
            text="Patient Folder:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.folder_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        self.folder_path_var = ctk.StringVar(value="No folder selected")
        self.folder_path_label = ctk.CTkLabel(
            self.folder_frame,
            textvariable=self.folder_path_var,
            font=ctk.CTkFont(size=12),
            text_color="gray",
            wraplength=600
        )
        self.folder_path_label.pack(anchor="w", padx=10, pady=(0, 10))
        
        self.select_button = ctk.CTkButton(
            self.folder_frame,
            text="📁 Select Patient Folder",
            font=ctk.CTkFont(size=14),
            height=40,
            command=self._select_folder
        )
        self.select_button.pack(pady=(0, 15), padx=10, fill="x")
        
        # Info about expected structure
        self.info_label = ctk.CTkLabel(
            self.folder_frame,
            text="Expected structure: PatientFolder/ → x10/ and x20/ subfolders with .tif images",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.info_label.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Analyze button
        self.analyze_button = ctk.CTkButton(
            scroll,
            text="🔬 Analyze Patient",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            fg_color="#27ae60",
            hover_color="#219a52",
            command=self._analyze,
            state="disabled"
        )
        self.analyze_button.pack(pady=15, padx=20, fill="x")
        
        # Progress section
        self.progress_frame = ctk.CTkFrame(scroll)
        self.progress_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_var = ctk.StringVar(value="Initializing classifier...")
        self.status_label = ctk.CTkLabel(
            self.progress_frame,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.set(0)
        
        # Results section
        self.results_frame = ctk.CTkFrame(scroll)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.results_title = ctk.CTkLabel(
            self.results_frame,
            text="Hierarchical Classification Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.results_title.pack(pady=(15, 10))
        
        # Binary prediction title
        self.binary_title = ctk.CTkLabel(
            self.results_frame,
            text="Level 1: Binary Classification (MF vs Non-MF)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="gray"
        )
        self.binary_title.pack(pady=(10, 5))
        
        # Prediction result
        self.prediction_var = ctk.StringVar(value="—")
        self.prediction_label = ctk.CTkLabel(
            self.results_frame,
            textvariable=self.prediction_var,
            font=ctk.CTkFont(size=32, weight="bold")
        )
        self.prediction_label.pack(pady=10)
        
        # Confidence
        self.confidence_var = ctk.StringVar(value="")
        self.confidence_label = ctk.CTkLabel(
            self.results_frame,
            textvariable=self.confidence_var,
            font=ctk.CTkFont(size=16)
        )
        self.confidence_label.pack(pady=5)
        
        # Multi-class prediction title
        self.multiclass_title = ctk.CTkLabel(
            self.results_frame,
            text="Level 2: Multi-Class Classification (5 Differential Diagnoses)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="gray"
        )
        self.multiclass_title.pack(pady=(15, 5))
        
        # Probability details - scrollable frame for 5 classes
        self.prob_scroll = ctk.CTkScrollableFrame(self.results_frame, fg_color="transparent")
        self.prob_scroll.pack(fill="x", padx=40, pady=15)
        
        # Create progress bars for each class
        self.class_progress_widgets = {}
        for i in range(5):  # 5 classes
            # Label
            class_label_var = ctk.StringVar(value=f"Class {i}:")
            class_label = ctk.CTkLabel(
                self.prob_scroll,
                textvariable=class_label_var,
                font=ctk.CTkFont(size=12)
            )
            class_label.grid(row=i, column=0, sticky="w", pady=5, padx=5)
            
            # Progress bar
            class_progress = ctk.CTkProgressBar(self.prob_scroll, width=300)
            class_progress.grid(row=i, column=1, padx=10, pady=5)
            class_progress.set(0)
            
            # Percentage label
            class_percent_var = ctk.StringVar(value="—")
            class_percent_label = ctk.CTkLabel(
                self.prob_scroll,
                textvariable=class_percent_var,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=60
            )
            class_percent_label.grid(row=i, column=2, pady=5, padx=5)
            
            # Store references
            self.class_progress_widgets[i] = {
                'label_var': class_label_var,
                'progress': class_progress,
                'percent_var': class_percent_var
            }
        
        # Save report button
        self.save_button = ctk.CTkButton(
            self.results_frame,
            text="📄 Save PDF Report",
            font=ctk.CTkFont(size=14),
            height=40,
            state="disabled",
            command=self._save_report
        )
        self.save_button.pack(pady=15)
        
        # Details text
        self.details_var = ctk.StringVar(value="")
        self.details_label = ctk.CTkLabel(
            self.results_frame,
            textvariable=self.details_var,
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.details_label.pack(pady=(0, 10))
    
        # ── Encoding maps (mirrors your original feature_mappings) ────────────────
    _FEATURE_MAPPINGS: dict[str, dict[str, int]] = {
        "sex":            {"Female": 0, "Male": 1},
        "course":         {
            "Intermittent": 0, "Progressive": 1, "Regressive": 2,
            "Remission And Excerbation": 3, "Stationary": 4, "Unknown": 5,
        },
        "color":          {
            "Erythematous": 0, "Hyperpigmented": 1,
            "Hypopigmented": 2, "Poikilodermatous": 3,
        },
        "bx1_site": {
            "Buttocks": 0, "Face": 1, "LL": 2, "Neck": 3,
            "No BX Site": 4, "Scalp": 5, "Trunk": 6, "UL": 7, "Unknown": 8,
        },
        "bx2_site": {
            "Buttocks": 0, "Face": 1, "LL": 2, "Neck": 3,
            "No BX Site": 4, "Trunk": 5, "UL": 6, "Unknown": 7,
        },
        "bx1_morph":  {"Macule": 0, "No BX Morph": 1, "Nodule": 2, "Papule": 3, "Patch": 4, "Plaque": 5},
        "bx2_morph":  {"Macule": 0, "No BX Morph": 1, "Nodule": 2, "Papule": 3, "Patch": 4, "Plaque": 5},
        "current_ttt":    {"No": 0, "Yes": 1},
        "visit_type":     {"FU": 0, "New": 1, "Rec": 2},
        "site_head_neck": {"No": 0, "Yes": 1},
        "site_ul":        {"No": 0, "Yes": 1},
        "site_ll":        {"No": 0, "Yes": 1},
        "site_trunk":     {"No": 0, "Yes": 1},
        "site_buttocks":  {"No": 0, "Yes": 1},
        "symptomatic":    {"No": 0, "Yes": 1},
        "macules":        {"No": 0, "Yes": 1},
        "patch":          {"No": 0, "Yes": 1},
        "papules":        {"No": 0, "Yes": 1},
        "plaque":         {"No": 0, "Yes": 1},
        "nodule":         {"No": 0, "Yes": 1},
        "scales":         {"No": 0, "Yes": 1},
    }
 
    # ── Logical sections (icon · title · ordered fields) ──────────────────────
    _FIELD_GROUPS = [
        {
            "icon":   "👤",
            "title":  "Patient Information",
            "fields": [
                ("age",             "Age (years)"),
                ("duration_months", "Duration (months)"),
                ("course",          "Disease Course"),
                ("visit_type",      "Visit Type"),
            ],
        },
        {
            "icon":   "📍",
            "title":  "Affected Sites",
            "fields": [
                ("site_head_neck", "Head & Neck"),
                ("site_ll",        "Lower Limbs"),
            ],
        },
        {
            "icon":   "🎨",
            "title":  "Lesion Characteristics",
            "fields": [
                ("color",   "Lesion Colour"),
                ("macules", "Macules"),
                ("patch",   "Patch"),
                ("papules", "Papules"),
                ("plaque",  "Plaque"),
                ("nodule",  "Nodule"),
                ("scales",  "Scales"),
            ],
        },
        {
            "icon":   "🔬",
            "title":  "Biopsy Information",
            "fields": [
                ("bx1_morph", "Biopsy 1 — Morphology"),
                ("bx2_morph", "Biopsy 2 — Morphology"),
                ("bx2_site",  "Biopsy 2 — Site"),
            ],
        },
    ]
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Tab builder (called by _create_widgets → _create_clinical_notes_tab)
    # ─────────────────────────────────────────────────────────────────────────
 
    def _create_clinical_notes_tab(self):
        """Build the refined Clinical Notes tab — replaces the original."""
        tab = self.tab_clinical
        self.clinical_inputs: dict = {}
        
        # Ensure tab expands to fill available space
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
 
        # ── Root scroll frame ──────────────────────────────────────────────
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent", corner_radius=0)
        scroll.grid(row=0, column=0, sticky="nsew")
        # Single stretchy column — cards will expand with the window
        scroll.columnconfigure(0, weight=1)
 
        # ── Page header ───────────────────────────────────────────────────
        self._cn_build_header(scroll)
 
        # ── One card per section ──────────────────────────────────────────
        for group in self._FIELD_GROUPS:
            self._cn_build_section_card(scroll, group)
 
        # ── Predict CTA ───────────────────────────────────────────────────
        self._cn_build_predict_button(scroll)
 
        # ── Results area (populated by show_clinical_result) ──────────────
        self.clinical_results_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        self.clinical_results_frame.pack(fill="both", expand=True, padx=20, pady=(12, 20))
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Public result display  — call this from _predict_clinical()
    # ─────────────────────────────────────────────────────────────────────────
 
    def show_clinical_result(
        self,
        diagnosis:  str,
        stage:      str | None  = None,
        confidence: float | None = None,
    ):
        """
        Render a styled result card beneath the predict button.
 
        Parameters
        ----------
        diagnosis   "MF" or "Non-MF"
        stage       Stage label if MF, e.g. "Patch-Plaque" or "Tumor"
        confidence  Probability from predict_proba (0.0–1.0), optional
        """
        # Clear previous result
        for w in self.clinical_results_frame.winfo_children():
            w.destroy()
 
        is_mf  = "mf" in diagnosis.lower()
        accent = _GREEN if is_mf else _SLATE
 
        # ── Card shell ────────────────────────────────────────────────────
        card = ctk.CTkFrame(
            self.clinical_results_frame,
            corner_radius=14,
            border_width=2,
            border_color=accent[0],
        )
        card.pack(fill="x", pady=(12, 0))
        card.columnconfigure(0, weight=1)
 
        # Coloured header strip
        hdr = ctk.CTkFrame(card, fg_color=accent, corner_radius=10)
        hdr.pack(fill="x", padx=6, pady=(6, 0))
        icon = "✅" if is_mf else "🔵"
        ctk.CTkLabel(
            hdr,
            text=f"  {icon}  Prediction Result",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="white",
            anchor="w",
        ).pack(fill="x", padx=10, pady=9)
 
        # Body grid: col-0 = labels, col-1 = values, col-2 = progress bar
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=16, pady=12)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=2)
 
        r = 0  # running row counter
 
        # Diagnosis
        self._cn_result_row(body, r, "Diagnosis", diagnosis, accent[0])
        r += 1
 
        # Confidence + progress bar
        if confidence is not None:
            self._cn_result_row(body, r, "Confidence", f"{confidence:.1%}", accent[0])
            pb = ctk.CTkProgressBar(
                body, height=10, corner_radius=5, progress_color=accent[0]
            )
            pb.grid(row=r, column=2, sticky="ew", padx=(10, 4), pady=6)
            pb.set(confidence)
            r += 1
 
        # Divider + stage (MF only)
        if is_mf and stage:
            ctk.CTkFrame(body, height=1, fg_color=_DIVIDER).grid(
                row=r, column=0, columnspan=3, sticky="ew", pady=(6, 8)
            )
            r += 1
            self._cn_result_row(body, r, "Stage of MF", stage, accent[0])
 
    # ─────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────────
 
    def _cn_build_header(self, parent: ctk.CTkScrollableFrame):
        """Page-level title and subtitle."""
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(16, 2))
 
        ctk.CTkLabel(
            hdr,
            text="Clinical Features Input",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        ).pack(fill="x")
 
        ctk.CTkLabel(
            hdr,
            text="Complete all sections, then press Predict to receive a diagnosis",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
        ).pack(fill="x", pady=(2, 0))
 
        # Thin rule
        ctk.CTkFrame(parent, height=1, fg_color=_DIVIDER).pack(
            fill="x", padx=20, pady=(8, 12)
        )
 
    def _cn_build_section_card(self, parent, group: dict):
        """
        One grouped card.
 
        Inner grid:
            col 0 = left label   (fixed width, left-anchored)
            col 1 = left widget  (stretches)
            col 2 = gutter
            col 3 = right label  (fixed width, left-anchored)
            col 4 = right widget (stretches)
        """
        # ── Card shell ────────────────────────────────────────────────────
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.pack(fill="x", padx=20, pady=(0, 10))
        card.columnconfigure(0, weight=1)
 
        # ── Coloured header strip ─────────────────────────────────────────
        hdr = ctk.CTkFrame(card, fg_color=_BLUE_DARK, corner_radius=8)
        hdr.pack(fill="x", padx=6, pady=(6, 0))
        ctk.CTkLabel(
            hdr,
            text=f"   {group['icon']}  {group['title']}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white",
            anchor="w",
        ).pack(fill="x", padx=6, pady=8)
 
        # ── Fields grid ───────────────────────────────────────────────────
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill="x", padx=6, pady=(6, 10))
 
        grid.columnconfigure(1, weight=1)   # left widget stretches
        grid.columnconfigure(2, minsize=20) # centre gutter
        grid.columnconfigure(4, weight=1)   # right widget stretches
 
        for idx, (feat, label) in enumerate(group["fields"]):
            is_left  = (idx % 2) == 0
            grid_row = idx // 2
            col_lbl  = 0 if is_left else 3
            col_wgt  = 1 if is_left else 4
 
            ctk.CTkLabel(
                grid,
                text=label + ":",
                font=ctk.CTkFont(size=11, weight="bold"),
                anchor="w",
                width=150,
            ).grid(row=grid_row, column=col_lbl, sticky="w",
                   padx=(12, 6), pady=7)
 
            widget = self._cn_make_widget(grid, feat)
            widget.grid(row=grid_row, column=col_wgt, sticky="ew",
                        padx=(0, 14), pady=7)
            self.clinical_inputs[feat] = widget
 
    def _cn_make_widget(self, parent, feature: str) -> ctk.CTkBaseClass:
        """Styled ComboBox for categoricals, Entry for numerics."""
        if feature in self._FEATURE_MAPPINGS:
            options = list(self._FEATURE_MAPPINGS[feature].keys())
            w = ctk.CTkComboBox(
                parent,
                values=options,
                font=ctk.CTkFont(size=11),
                state="readonly",
                height=34,
                button_color=_BLUE_MID,
                border_color=_BLUE_MID,
                dropdown_hover_color=_BLUE_MID,
            )
            w.set(options[0])
        else:
            w = ctk.CTkEntry(
                parent,
                placeholder_text="Enter value",
                font=ctk.CTkFont(size=11),
                height=34,
                border_color=_BLUE_MID,
            )
        return w
 
    def _cn_build_predict_button(self, parent):
        """Full-width green CTA button."""
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", padx=20, pady=(6, 12))
        ctk.CTkButton(
            wrap,
            text="🔍  Predict Diagnosis",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=48,
            corner_radius=12,
            fg_color=_GREEN,
            hover_color=_GREEN[1],
            command=self._predict_clinical,
        ).pack(fill="x")
 
    @staticmethod
    def _cn_result_row(parent, row: int, label: str, value: str, color: str):
        """Single label + value pair inside the result card."""
        ctk.CTkLabel(
            parent,
            text=label + ":",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
            width=120,
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=5)
 
        ctk.CTkLabel(
            parent,
            text=value,
            font=ctk.CTkFont(size=12),
            text_color=color,
            anchor="w",
        ).grid(row=row, column=1, sticky="w", pady=5)

    
    def _init_classifier_async(self):
        """Initialize classifier in background thread."""
        def init():
            try:
                self.classifier = MFClassifier(device='cpu')
                self.classifier.load_models(
                    progress_callback=lambda msg: self._update_status(msg)
                )
                self._update_status("Ready! Select a patient folder to analyze.")
                self.after(0, lambda: self.analyze_button.configure(state="normal"))
            except Exception as e:
                self._update_status(f"Error loading models: {str(e)}")
                self.after(0, lambda: messagebox.showerror("Error", f"Failed to load models:\n{str(e)}"))
        
        threading.Thread(target=init, daemon=True).start()
    
    def _update_status(self, message: str):
        """Update status label (thread-safe)."""
        self.after(0, lambda: self.status_var.set(message))
    
    def _select_folder(self):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(
            title="Select Patient Folder",
            mustexist=True
        )
        
        if folder:
            self.selected_folder = Path(folder)
            self.folder_path_var.set(str(self.selected_folder))
            
            # Validate folder structure
            from preprocessing import validate_patient_folder
            is_valid, message = validate_patient_folder(self.selected_folder)
            
            if is_valid:
                self.status_var.set(f"✓ {message}")
                self.analyze_button.configure(state="normal")
            else:
                self.status_var.set(f"⚠ {message}")
                self.analyze_button.configure(state="disabled")
                messagebox.showwarning("Invalid Folder", message)
    
    def _analyze(self):
        """Run analysis on selected folder."""
        if not self.selected_folder or not self.classifier:
            return
        
        # Disable buttons during analysis
        self.analyze_button.configure(state="disabled")
        self.select_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        
        # Reset results
        self.prediction_var.set("Analyzing...")
        self.confidence_var.set("")
        
        # Reset all class progress bars
        for i in range(5):
            self.class_progress_widgets[i]['progress'].set(0)
            self.class_progress_widgets[i]['percent_var'].set("—")
        
        self.details_var.set("")
        self.progress_bar.set(0)
        
        def run_analysis():
            try:
                # Run prediction
                results = self.classifier.predict(
                    self.selected_folder,
                    progress_callback=lambda msg: self._update_status(msg)
                )
                
                self.current_results = results
                
                # Update UI with results
                self.after(0, lambda: self._display_results(results))
                
            except Exception as e:
                self.after(0, lambda: self._handle_error(str(e)))
            finally:
                self.after(0, self._enable_buttons)
        
        threading.Thread(target=run_analysis, daemon=True).start()
    
    def _display_results(self, results: dict):
        """Display prediction results in UI with hierarchical classification."""
        # Binary level prediction
        binary_pred = results['binary_prediction']
        binary_conf = results['binary_confidence'] * 100
        mf_prob = results['mf_probability'] * 100
        non_mf_prob = results['non_mf_probability'] * 100
        
        # Multi-class level prediction
        prediction = results['predicted_class'] 
        confidence = results['confidence'] * 100

        if binary_pred == "Non-MF" and prediction == "MF":
            # If binary says Non-MF but multi-class says MF, we take the second highest class
            class_probs = results['class_probabilities']
            sorted_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            # Find the second highest class that is not MF
            for class_name, prob in sorted_classes:
                if class_name != "MF":
                    prediction = class_name
                    confidence = prob * 100
                    break   
        
        # Color for binary prediction
        binary_color = "#e74c3c" if binary_pred == "MF" else "#3498db"  # Red for MF, Blue for Non-MF
        
        # Display Binary Prediction (Level 1)
        binary_text = f"{binary_pred}\nBinary Confidence: {binary_conf:.1f}%"
        self.prediction_var.set(binary_text)
        self.prediction_label.configure(text_color=binary_color)
        
        # Display Binary Accuracy as (MF%, Non-MF%)
        self.confidence_var.set(f"Binary Accuracy: ({mf_prob:.1f}% MF, {non_mf_prob:.1f}% Non-MF)")
        
        # Display all class probabilities
        class_probs = results['class_probabilities']
        
        # Define colors for each class
        class_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
        
        for i in range(5):
            # Get actual class name from results
            class_name = list(class_probs.keys())[i]
            prob = class_probs[class_name]
            
            # Update label with full class name
            self.class_progress_widgets[i]['label_var'].set(f"{class_name}:")
            
            # Update progress bar
            self.class_progress_widgets[i]['progress'].set(prob)
            self.class_progress_widgets[i]['progress'].configure(progress_color=class_colors[i])
            
            # Update percentage - highlight the predicted class
            percentage_text = f"{prob*100:.1f}%"
            if class_name == prediction:
                percentage_text += " ✓"  # Mark the predicted class
            self.class_progress_widgets[i]['percent_var'].set(percentage_text)
        
        # Details - Include both binary and multi-class info
        details = (
            f"Binary Prediction: {binary_pred} ({binary_conf:.1f}%) | "
            f"Specific Class: {prediction} ({confidence:.1f}%)\n"
            f"x10: {results['n_x10_patches']} patches from {results['n_x10_images']} images | "
            f"x20: {results['n_x20_patches']} patches from {results['n_x20_images']} images | "
            f"Fusion weight: {results['fusion_weight']:.2f}"
        )
        self.details_var.set(details)
        
        # Enable save button
        self.save_button.configure(state="normal")
        self.progress_bar.set(1)
        self.status_var.set("✓ Analysis complete!")
    
    def _handle_error(self, error_message: str):
        """Handle analysis error."""
        self.prediction_var.set("Error")
        self.prediction_label.configure(text_color="gray")
        self.status_var.set(f"Error: {error_message}")
        messagebox.showerror("Analysis Error", error_message)
    
    def _enable_buttons(self):
        """Re-enable buttons after analysis."""
        self.analyze_button.configure(state="normal")
        self.select_button.configure(state="normal")
    
    def _predict_clinical(self):
        """Run clinical features prediction."""
        try:
            # Get input values
            features = ['macules', 'bx1_morph', 'papules', 'age', 'color', 'patch',
                       'bx2_morph', 'visit_type', 'duration_months', 'plaque', 'scales',
                       'site_head_neck', 'site_ll', 'course', 'bx2_site', 'nodule']
            
            feature_values = []
            for feature in features:
                widget = self.clinical_inputs[feature]
                value_str = widget.get().strip()
                
                if not value_str:
                    messagebox.showerror("Error", f"Please select/enter a value for {feature}")
                    return
                
                try:
                    # Check if this is a categorical feature with mapping
                    if feature in self._FEATURE_MAPPINGS:
                        # Map the dropdown selection to numeric value
                        value = self._FEATURE_MAPPINGS[feature][value_str]
                    else:
                        # Parse numeric value
                        value = float(value_str)
                    
                    feature_values.append(value)
                except (ValueError, KeyError) as e:
                    messagebox.showerror("Error", f"Invalid value for {feature}.")
                    return
            
            def run_prediction():
                try:
                    # Create DataFrame with features
                    X = pd.DataFrame([feature_values], columns=features)
                    
                    # Get model paths
                    models_dir = Path(__file__).parent / "models"
                    diagnosis_model_path = models_dir / "rf_model_diagnosis_mf.pkl"
                    stage_model_path = models_dir / "rf_model_stage_mf.pkl"
                    
                    # Load and run diagnosis model
                    diagnosis_bundle = joblib.load(diagnosis_model_path)
                    X_diag = X[diagnosis_bundle['features']]
                    X_imp_diag = diagnosis_bundle['imputer'].transform(X_diag)
                    y_pred_enc = diagnosis_bundle['model'].predict(X_imp_diag)
                    y_prob = diagnosis_bundle['model'].predict_proba(X_imp_diag)
                    y_pred_labels = diagnosis_bundle['label_encoder'].inverse_transform(y_pred_enc)
                    
                    diagnosis = y_pred_labels[0]
                    diagnosis_prob = y_prob[0][y_pred_enc[0]]
                    
                    stage = None
                    
                    # If diagnosis is MF, run stage model
                    if diagnosis == "MF":
                        stage_bundle = joblib.load(stage_model_path)
                        X_stage = X[stage_bundle['features']]
                        X_imp_stage = stage_bundle['imputer'].transform(X_stage)
                        y_pred_enc_stage = stage_bundle['model'].predict(X_imp_stage)
                        y_pred_labels_stage = stage_bundle['label_encoder'].inverse_transform(y_pred_enc_stage)
                        stage = y_pred_labels_stage[0]
                    
                    # Display result using show_clinical_result
                    self.after(0, lambda d=diagnosis, s=stage, c=diagnosis_prob: self.show_clinical_result(d, s, c))
                    
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Prediction Error", f"Error during prediction:\n{str(e)}"))
                    self.after(0, lambda: self.clinical_diag_var.set("Error"))
            
            threading.Thread(target=run_prediction, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing input:\n{str(e)}")
    
    def _save_report(self):
        """Save PDF report."""
        if not self.current_results:
            return
        
        # Default filename
        patient_name = self.current_results['patient_name']
        default_filename = f"{patient_name}_MF_Report.pdf"
        
        # Ask for save location
        filepath = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=default_filename
        )
        
        if filepath:
            try:
                output_path = generate_report(self.current_results, Path(filepath))
                self.status_var.set(f"✓ Report saved: {output_path.name}")
                messagebox.showinfo("Success", f"Report saved successfully!\n\n{output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report:\n{str(e)}")
               


def main():
    """Entry point for the application."""
    app = MFClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
