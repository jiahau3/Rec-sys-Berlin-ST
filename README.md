# Rec-sys-Berlin-ST
How to run the application on your local station:

1. Download: 3_task_4_no_duplicates_reduced.zip
2. Use sentence_transformers_model.ipynb to create: final_matrix_f64_noP.npy and similarity_top_k.npy ( will take a long time to produce, around 1h )
   OR
   Download files from:
   https://huggingface.co/spaces/jiahau/Rec-sys-Berlin-ST/tree/main
3. Install Stremlit
4. In console, navigate to the current folder and run:
	streamlit run app.py
