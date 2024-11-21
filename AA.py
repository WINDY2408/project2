import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import threading
import queue
from datetime import datetime
import torch.nn as nn
from torchvision import transforms, models
import time
import winsound  # For Windows alert sound
import os

class DistractedDriverModel(nn.Module):
    """Model architecture using EfficientNet-B0"""
    def __init__(self, num_classes=10):
        super(DistractedDriverModel, self).__init__()
        self.base_model = models.efficientnet_b0(weights=None)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

class DistractedDriverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Distracted Driver Detection")
        self.root.geometry("1200x800")
        
        # Variables
        self.is_running = False
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.current_source = None
        self.cap = None
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.get_transforms()
        
        # Class labels
        self.classes = {
            0: "Safe Driving",
            1: "Texting - Right",
            2: "Talking on Phone - Right",
            3: "Texting - Left",
            4: "Talking on Phone - Left",
            5: "Operating Radio",
            6: "Drinking",
            7: "Reaching Behind",
            8: "Hair and Makeup",
            9: "Talking to Passenger"
        }
        
        self.setup_gui()
        self.setup_alerts()
        
    def load_model(self):
        try:
            model = DistractedDriverModel().to(self.device)
            model_path = r"F:\car\models\20241028_025819\best_model_98_val_accuracy=1.0000.pt" # Update with your model path
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", "Model file not found!")
                return None
                
            state_dict = torch.load(model_path, map_location=self.device)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            return None
            
    def get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def setup_gui(self):
        # Create frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Load Image", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Load Video", 
                  command=self.load_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Start Camera", 
                  command=self.start_camera).pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", 
                                    command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Display area
        self.canvas = tk.Canvas(self.display_frame, bg='black')
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        # Results areag
        self.result_frame = ttk.Frame(self.display_frame)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        self.result_label = ttk.Label(self.result_frame, text="Prediction: None",
                                    font=('Arial', 12, 'bold'))
        self.result_label.pack(pady=10)
        
        # Confidence bars
        self.confidence_bars = {}
        self.confidence_labels = {}
        for i, class_name in self.classes.items():
            frame = ttk.Frame(self.result_frame)
            frame.pack(fill=tk.X, pady=2)
            
            self.confidence_labels[i] = ttk.Label(frame, text=f"{class_name}: 0%",
                                                width=25)
            self.confidence_labels[i].pack(side=tk.LEFT)
            
            self.confidence_bars[i] = ttk.Progressbar(frame, length=150, mode='determinate')
            self.confidence_bars[i].pack(side=tk.LEFT, padx=5)
            
    def setup_alerts(self):
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds
        self.alert_threshold = 0.8  # 80% confidence
        
    def play_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
            self.last_alert_time = current_time
            
    def process_frame(self, frame):
        try:
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Transform and predict
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
            # Get prediction
            pred_idx = torch.argmax(probabilities).item()
            pred_prob = probabilities[pred_idx].item()
            
            # Update confidence bars
            for i, prob in enumerate(probabilities):
                prob_value = prob.item() * 100
                self.confidence_bars[i]['value'] = prob_value
                self.confidence_labels[i].config(
                    text=f"{self.classes[i]}: {prob_value:.1f}%"
                )
                
            # Update prediction label
            self.result_label.config(
                text=f"Prediction: {self.classes[pred_idx]}\nConfidence: {pred_prob*100:.1f}%"
            )
            
            # Check for alerts
            if pred_idx != 0 and pred_prob > self.alert_threshold:  # Not safe driving
                self.play_alert()
                
            return frame
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame
            
    def update_frame(self):
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is not None:
                    # Process frame for display
                    frame = cv2.resize(frame, (800, 600))
                    photo = ImageTk.PhotoImage(image=Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    
                    # Update canvas
                    self.canvas.config(width=photo.width(), height=photo.height())
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                    self.canvas.image = photo  # Keep a reference
                    
            if self.is_running:
                self.root.after(10, self.update_frame)
                
        except Exception as e:
            print(f"Error updating frame: {str(e)}")
            
    def process_video(self):
        try:
            while self.is_running and self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                if self.frame_queue.full():
                    self.frame_queue.get()  # Remove oldest frame
                self.frame_queue.put(processed_frame)
                
            self.cap.release()
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            self.stop_processing()
            
    def load_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
            )
            if file_path:
                self.stop_processing()
                frame = cv2.imread(file_path)
                if frame is not None:
                    self.is_running = True
                    self.stop_button.config(state=tk.NORMAL)
                    processed_frame = self.process_frame(frame)
                    self.frame_queue.put(processed_frame)
                    self.update_frame()
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
            
    def load_video(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
            if file_path:
                self.stop_processing()
                self.cap = cv2.VideoCapture(file_path)
                if self.cap.isOpened():
                    self.is_running = True
                    self.stop_button.config(state=tk.NORMAL)
                    threading.Thread(target=self.process_video, daemon=True).start()
                    self.update_frame()
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
            
    def start_camera(self):
        try:
            self.stop_processing()
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.stop_button.config(state=tk.NORMAL)
                threading.Thread(target=self.process_video, daemon=True).start()
                self.update_frame()
            else:
                messagebox.showerror("Error", "Could not open camera!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error starting camera: {str(e)}")
            
    def stop_processing(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.stop_button.config(state=tk.DISABLED)
        
    def on_closing(self):
        self.stop_processing()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DistractedDriverGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()