from vultrition.models.results import AnalysisResults



def create_label_url(results : AnalysisResults) -> None:
    base64_encoded_results = results.to_base64()