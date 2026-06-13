from pathlib import Path
from data_ingestion import DataIngestion
from train import ModelTrainer
from evaluation import ModelEvaluator

class CreditScorePipeline:
    """Pipeline utama untuk orkestrasi data, pelatihan, evaluasi, dan pemilihan model terbaik."""
    
    def __init__(self, raw_data_path: str | Path, accuracy_threshold: float = 0.70):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)
        self.ingested_dir = self.base_dir / "ingested"
        self.accuracy_threshold = accuracy_threshold
        
        # Inisialisasi komponen kelas yang sudah kita buat
        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print("Mengeksekusi Pipeline Klasifikasi Skor Kredit...")
        
        # 1. Tarik dan simpan data mentah
        ingested_file_path = self.ingestor.run()
        
        # Variabel untuk melacak model terbaik untuk keperluan Streamlit nanti
        best_model_name = None
        best_accuracy = 0.0
        
        print("\n*** Memulai Pelatihan Multi-Model ***")
        
        # 2 & 3. Loop melalui semua model yang ada di dalam dictionary trainer
        for model_name in self.trainer.models.keys():
            print(f"\nMemproses Model: {model_name.upper()}")
            
            # Proses Latih (Training)
            run_id, x_test, y_test = self.trainer.run(ingested_file_path, model_name)
            
            # Proses Evaluasi (menerima 4 nilai kembalian termasuk f1_score)
            accuracy, precision, recall, f1 = self.evaluator.run(run_id, model_name, x_test, y_test)
            
            # 4. Penilaian kelayakan model
            if accuracy >= self.accuracy_threshold:
                print(f"Sukses: Model memenuhi standar kualitas (Akurasi: {accuracy:.3f})")
            else:
                print(f"Ditolak: Akurasi model ({accuracy:.3f}) di bawah batas ({self.accuracy_threshold})")
            
            # Membandingkan tingkat akurasi untuk mencari model yang paling unggul
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                
        # Kesimpulan akhir pipeline
        print(f"Model terbaik yang siap ditarik ke Streamlit adalah: {best_model_name.upper()} (Akurasi: {best_accuracy:.3f})")


if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_A.csv"
    
    credit_pipeline = CreditScorePipeline(raw_data_path=DATA_INPUT, accuracy_threshold=0.75)
    credit_pipeline.execute()