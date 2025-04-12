from enum import Enum

class DataEngine(Enum):
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"
    LLM = "Language Models"
    
    def get_features(self):
        if self == DataEngine.FINANCE:
            return {
                'amount': True,
                'transaction_type': True,
                'merchant_category': True,
                'bank_type': True,
                'city': True,
                'customer_age': True,
                'customer_tenure': True,
                'transaction_frequency': True,
                'credit_score': True,
                'is_fraud': True
            }
        elif self == DataEngine.HEALTHCARE:
            return {
                'patient_age': True,
                'gender': True,
                'blood_pressure': True,
                'heart_rate': True,
                'temperature': True,
                'diagnosis': True,
                'medication': True,
                'admission_type': True,
                'length_of_stay': True,
                'insurance_type': True
            }
        elif self == DataEngine.LLM:
            return {
                'prompt_length': True,
                'response_length': True,
                'language': True,
                'topic': True,
                'sentiment': True,
                'complexity': True,
                'format_type': True,
                'model_type': True,
                'temperature': True,
                'top_p': True
            }
        return {}