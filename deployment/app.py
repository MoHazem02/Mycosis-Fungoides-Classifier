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

# Add current directory to path for imports when bundled
if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)
    sys.path.insert(0, sys._MEIPASS)

from classifier import MFClassifier
from report_generator import generate_report


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
        
        # Main container
        self.main_frame = ctk.CTkScrollableFrame(self)
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
        
        # Folder selection frame
        self.folder_frame = ctk.CTkFrame(self.main_frame)
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
            self.main_frame,
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
        self.progress_frame = ctk.CTkFrame(self.main_frame)
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
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.results_title = ctk.CTkLabel(
            self.results_frame,
            text="Classification Results",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.results_title.pack(pady=(15, 10))
        
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
        
        # Probability details
        self.prob_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.prob_frame.pack(fill="x", padx=40, pady=15)
        
        # MF probability bar
        self.mf_label = ctk.CTkLabel(
            self.prob_frame,
            text="MF Probability:",
            font=ctk.CTkFont(size=12)
        )
        self.mf_label.grid(row=0, column=0, sticky="w", pady=5)
        
        self.mf_progress = ctk.CTkProgressBar(self.prob_frame, width=300)
        self.mf_progress.grid(row=0, column=1, padx=10, pady=5)
        self.mf_progress.set(0)
        
        self.mf_percent_var = ctk.StringVar(value="—")
        self.mf_percent_label = ctk.CTkLabel(
            self.prob_frame,
            textvariable=self.mf_percent_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60
        )
        self.mf_percent_label.grid(row=0, column=2, pady=5)
        
        # Non-MF probability bar
        self.nonmf_label = ctk.CTkLabel(
            self.prob_frame,
            text="Non-MF Probability:",
            font=ctk.CTkFont(size=12)
        )
        self.nonmf_label.grid(row=1, column=0, sticky="w", pady=5)
        
        self.nonmf_progress = ctk.CTkProgressBar(self.prob_frame, width=300)
        self.nonmf_progress.grid(row=1, column=1, padx=10, pady=5)
        self.nonmf_progress.set(0)
        
        self.nonmf_percent_var = ctk.StringVar(value="—")
        self.nonmf_percent_label = ctk.CTkLabel(
            self.prob_frame,
            textvariable=self.nonmf_percent_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=60
        )
        self.nonmf_percent_label.grid(row=1, column=2, pady=5)
        
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
        self.mf_progress.set(0)
        self.nonmf_progress.set(0)
        self.mf_percent_var.set("—")
        self.nonmf_percent_var.set("—")
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
        """Display prediction results in UI."""
        # Update prediction label
        prediction = results['predicted_class']
        if prediction == "MF":
            self.prediction_var.set("🔴 MYCOSIS FUNGOIDES")
            self.prediction_label.configure(text_color="#e74c3c")
        else:
            self.prediction_var.set("🟢 NON-MF")
            self.prediction_label.configure(text_color="#27ae60")
        
        # Confidence
        confidence = results['confidence'] * 100
        self.confidence_var.set(f"Confidence: {confidence:.1f}%")
        
        # Probability bars
        mf_prob = results['mf_probability']
        nonmf_prob = results['nonmf_probability']
        
        self.mf_progress.set(mf_prob)
        self.nonmf_progress.set(nonmf_prob)
        self.mf_percent_var.set(f"{mf_prob*100:.1f}%")
        self.nonmf_percent_var.set(f"{nonmf_prob*100:.1f}%")
        
        # Color the progress bars
        if mf_prob > nonmf_prob:
            self.mf_progress.configure(progress_color="#e74c3c")
            self.nonmf_progress.configure(progress_color="#95a5a6")
        else:
            self.mf_progress.configure(progress_color="#95a5a6")
            self.nonmf_progress.configure(progress_color="#27ae60")
        
        # Details
        details = (
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
