# Neurova

Neurova is a small web app where two deep learning models look at brain data and a clinician checks what they say.

The MRI model is an EfficientNet-B0 that sorts a brain MRI slice into glioma, meningioma, pituitary tumour or no tumour, and draws a Grad-CAM heatmap of where it looked. It gets 98.8% on 924 test scans it never saw during training.

The sleep model is a 1-D CNN with a bidirectional GRU. It labels each 30 seconds of an overnight EEG recording as Wake, N1, N2, N3 or REM, and gets 83.5% (kappa 0.77) on two people it never saw.

There are three kinds of account. Patients upload files and choose which clinicians can see their results. Clinicians work from a triage list of results waiting for review, with likely tumours at the top, and record whether they agree with the model. Admins manage accounts and read the activity log, and can't open anyone's results.

I built this for the Deep Learning course (CDS301) at Sai University. It's a student project for learning and research, so its results are not a medical diagnosis.

## Running it

You need Python 3.10 to 3.13 from [python.org](https://www.python.org/downloads/). On Windows, tick "Add python.exe to PATH" in the installer.

On Windows, double-click `start.bat` or run `.\start.bat` in the VS Code terminal. On macOS or Linux, run `./start.sh`.

The first start installs what it needs (about 100 MB) and takes a minute or two. After that it starts in a few seconds and opens http://localhost:8000.

The sign-in page has buttons for the demo accounts. They all use the password `Neurova-demo-2026`.

## Trying it out

1. Sign in as Akshata Nagaraj, a patient. She already has two results, a brain MRI and a sleep recording. Open both, and download the PDF report for one of them.
2. Go to New analysis and pick an example scan. The result comes back in under a second. The two examples with an amber label are scans the model gets wrong. The first one is a missed tumour, and its result page shows high uncertainty, a failed stability check and an "unfamiliar scan" warning.
3. The Sharing page shows which clinicians can see her results.
4. Sign in as Dr. Aahana Sharma, a clinician. Her Triage page lists likely tumours first, then results the model is unsure about (the missed tumour is in this group), then routine ones. Open Akshata's glioma result, save a review, and you go back to the list.
5. Sign in as Akshata again and the clinician's note is on her result.
6. Sign in as Site Administrator to see the accounts and the activity log. Edit lets you rename an account, reset its password or disable it.
7. Any account can open How the models perform, which has the test results, including the cases the models get wrong.

You can upload your own files too: a brain MRI slice as PNG or JPG, or an EDF sleep recording at least five minutes long.

## How it's put together

```
browser  (web/)  ──►  server  (app/server.py)  ──►  models  (models/*.onnx, run by ONNX Runtime)
                          │
                          └──►  database  (data/neurova.db, created on first run)
```

I trained the models in PyTorch and exported them to ONNX, so the app only needs ONNX Runtime to run them. ONNX Runtime is small, while PyTorch is about 1 GB.

- `app/server.py` has every address the browser can call: accounts, sharing, analyses and admin.
- `app/mri.py` does the MRI analysis: the prediction, Grad-CAM, uncertainty, the stability check and the unfamiliar-scan check.
- `app/sleep.py` does the sleep staging, draws the hypnogram and compares it with the expert's scoring.
- `app/preprocess.py` prepares scans and recordings. The training scripts import this same file.
- `app/edf.py` reads EDF recordings.
- `app/auth.py` handles passwords, sign-in tokens and the rule for who can see which patient.
- `app/database.py` defines the four tables: users, access, studies and audit.
- `app/report_pdf.py` makes the one-page PDF report (its font is in `app/fonts/`).
- `app/demo.py` creates the made-up demo accounts and their results.
- `web/` holds the pages (`index.html`, `style.css`, `app.js`), with no framework or build step.
- `models/` holds the trained models as ONNX files (17 MB).
- `samples/` has 22 test scans, two of which the model gets wrong, and one sleep recording. None of them were used in training.
- `results/` has the figures and numbers from the test set, which the model page shows.
- `training/` has the PyTorch training code and the ONNX export.
- `tests/` has automated checks for the server and the pages.
- `report/` has the CDS301 report, slides, business model canvas and viva notes.

## The models

MRI: the public Brain Tumor MRI Dataset has 7,023 images. Many images appeared in both its official training and testing folders, which would have inflated the score, so I removed 866 duplicates before splitting. I then trained three models on a new 70/15/15 split:

| Model | Test accuracy | Predictions changed by faint noise |
| --- | --- | --- |
| Small CNN from scratch | 89.3% | 36.5% |
| EfficientNet-B0, standard fine-tuning | 98.8% | 35.1% |
| EfficientNet-B0, fingerprint-scrub training (in use) | 98.8% | 0.6% |

When I added noise too faint to see, the standard model changed its answer on about a third of the test scans. That suggests it had partly learned which source collection each image came from. Training with augmentation that wipes out those fingerprints (resampling, JPEG re-compression, blur and noise) fixed this and kept the same accuracy.

Sleep: the sleep model uses 20 nights from 10 healthy adults in Sleep-EDF Expanded, with one EEG channel (Fpz-Cz). I split the data by person into training, validation and test sets, so the test score is for people the model has never seen. It does worst on N1, and it sometimes labels quiet wakefulness as N1.

I also tried swapping the GRU for a Transformer encoder and kept everything else the same. Its first run scored 85.3%, higher than the GRU's 83.5%. With only two test people one run can be lucky, so I trained each model three times with different random seeds. The GRU averaged 84.7% (83.5 to 86.8%) and the Transformer 77.0% (66.0 to 85.3%), so the app uses the GRU. `python training/compare_sleep.py` reruns the comparison.

The MRI model only knows four answers, all of them tumours inside the brain, and it has to pick one even when the scan shows something else. When I showed it an eye-socket tumour I found online, it said meningioma or no tumour. The app therefore measures how far each scan is from the training scans and warns when one looks unfamiliar. On the test set this check flags 8 of the model's 11 mistakes and 3.5% of the scans it gets right, so it can still miss a bad case.

Neither model has been tested on data from a hospital outside these public datasets.

## Retraining

You don't need to retrain anything to run the app. Training uses PyTorch and runs on a CPU.

```
pip install -r requirements.txt -r training/requirements.txt
python training/get_data.py --sleep     # downloads the datasets into datasets/
python training/train_mri.py            # about 2 hours
python training/train_sleep.py          # about 5 minutes
python training/export_onnx.py          # converts to ONNX and updates models/, results/, samples/
python training/fit_familiarity.py      # refits the unfamiliar-scan check (2 minutes)
python training/compare_sleep.py        # optional: GRU vs Transformer, three seeds each (30 minutes)
```

`training/Train_in_Colab.ipynb` runs the same steps in Google Colab.

## Tests

```
python tests/test_app.py        # 101 checks of the server (needs: pip install httpx)
python tests/test_browser.py    # clicks through every page (needs: pip install playwright, then playwright install chromium)
```

## Live version

Neurova is live at https://YOUR-DOMAIN, on my own domain. It runs on a Linux VPS managed with CloudPanel, which handles the domain, the HTTPS certificate (Let's Encrypt) and an Nginx reverse proxy in front of the app.

The setup, if you want to host your own copy:

1. In CloudPanel, add a Python site for your domain and give it an app port, for example 8000. CloudPanel sets up Nginx to forward the domain to that port.
2. Turn on a Let's Encrypt certificate for the site in its SSL/TLS tab.
3. Over SSH, clone this repo into the site's folder, create a virtual environment and install the requirements:
   ```
   git clone https://github.com/YOUR-USERNAME/neurova.git
   cd neurova
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
4. Start the app on the port you chose. Setting `PORT` makes it listen on all interfaces and skip opening a browser:
   ```
   PORT=8000 NEUROVA_SECRET=some-long-random-string .venv/bin/python run.py
   ```
   To keep it running after you log out and restart it after a reboot, run that command as a systemd service.

Set `NEUROVA_SECRET` on a public server so sign-in tokens use a key you control, and `NEUROVA_DEMO=0` if you don't want the demo accounts. The database lives in `data/` and stays there between restarts.

The `Dockerfile` also works if you'd rather run it in a container.

## Other notes

The data comes from the Brain Tumor MRI Dataset (M. Nickparvar, 2021, CC BY 4.0) and Sleep-EDF Expanded (B. Kemp et al., 2000, via PhysioNet, Open Data Commons Attribution). The EfficientNet-B0 ImageNet weights come from EfficientNet-PyTorch (Apache 2.0).

To use MySQL instead of SQLite, run `pip install pymysql cryptography`, create an empty database, and set `DATABASE_URL=mysql+pymysql://user:password@localhost/neurova` before starting.

To rename the demo people, change the names in `app/demo.py` (leave the emails alone), delete the `data` folder and start again. Deleting `data` on its own also gives you a fresh start, and the demo accounts come back the next time it runs.

The app doesn't keep uploaded files. It stores the result, which for an MRI includes the cropped image and its heatmap. Passwords are stored as bcrypt hashes.
