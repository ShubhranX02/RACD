#!/bin/bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
source venv/bin/activate
cd src
exec streamlit run dashboard.py --server.headless true --server.port 8501
