# ANP
PADS is a Phase 1 malware detection prototype that identifies post-attack threats using machine learning, signature scanning, and log analysis. It collects system data, detects suspicious behavior, and provides a foundation for advanced security features in future phases.

The Post-Attack Detection System (PADS) is a Phase 1 malware detection prototype designed to identify threats after an attack has occurred. It combines machine learning, signature-based scanning, and log analysis to detect suspicious activity, classify unknown samples, and provide a foundation for advanced security features in future phases. This phase focuses on core components such as feature extraction, baseline ML models, dataset processing, and a functional detection pipeline.

ANP/
├── samples/
│   ├── benign/                
│   └── malware/              
│
├── model_pipeline.joblib    
│
├── app.py                   
├── extract_features.py      
├── model.py                  
├── Random_forest.py                           
│
├── Final_Dataset_without_duplicate.csv   
├── new_sample.csv             
└── new.py  

The project includes:
Cleaned dataset: Final_Dataset_without_duplicate.csv
Sample inputs for testing detection: new_sample.csv


