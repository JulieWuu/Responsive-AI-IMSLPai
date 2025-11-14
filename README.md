# Responsive AI - IMSLPai

**Responsive AI** based on `google.generativeai` with a **web scraper** helper that fetches data from the IMSLP website by composers. 

This project was developed within 3 days with the help of generative AI. It aims to provide classical musicians with guidance to piece choices, composer and time period information. <br><br>

Main AI file `IMSLPai.py`, web scraper `scrapeSelenium.py`. 

The csv file `premium_data.csv` was obtained by running `scrapeSelenium.py` repeatedly on many different composers of the user's choice. When running `IMSLPai.py`, the csv file specified (in this case `premium_data.csv`) is turned into the database `imslp.db` that `google.generativeai` can understand.
