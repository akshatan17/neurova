# Neurova

A small web app in which two deep learning models analyse brain data, and a clinician reviews what they say.

- **Brain MRI.** EfficientNet-B0 sorts an MRI slice into glioma, meningioma, pituitary tumour or no tumour, and shows a Grad-CAM heatmap of where it looked. Test accuracy 98.8% on 924 scans it never saw.
- **Sleep EEG.** A 1-D CNN with a bidirectional GRU labels every 30 seconds of an overnight recording as Wake, N1, N2, N3 or REM. Test accuracy 83.5% (kappa 0.77) on two people it never saw.

There are three kinds of account. A **patient** uploads files and chooses which clinicians may see the results. A **clinician** works from a triage list of every result waiting for review, likely tumours first, and records whether they agree with the model. An **admin** manages accounts and reads the activity log, but cannot open anyone's results.

Built for the Deep Learning course CDS301 at Sai University. It is a student project for education and research, and its results are not a medical diagnosis.

## Run it

You need Python 3.10 to 3.13 from [python.org](https://www.python.org/downloads/). On Windows, tick "Add python.exe to PATH" in the installer.

**Windows:** double-click `start.bat`, or type `.\start.bat` in the VS Code terminal.
**macOS or Linux:** `./start.sh`

The first start sets things up, which takes a minute or two (about 100 MB of downloads). After that it starts in seconds and opens http://localhost:8000 in your browser.

On the sign-in page, click one of the demo accounts. The password for all of them is `Neurova-demo-2026`.

## A two-minute tour

1. Sign in as **Akshata Nagaraj** (patient). Two results are waiting: a brain MRI and a sleep recording. Open each one, and click **Download PDF report** on one.
2. Click **New analysis** and pick any example scan. The result appears in under a second. The two scans with an amber label are ones the model gets wrong: open the first to see high uncertainty, a failed stability check and an "unfamiliar scan" warning on a missed tumour.
3. Open **Sharing** to see which clinicians can view the results.
4. Sign out and sign in as **Dr. Aahana Sharma** (clinician). The **Triage** page lists every result waiting for her, likely tumours first, then results the model is unsure about (including the missed tumour), then routine ones. Open Akshata's glioma, save a review, and you land back on the list.
5. Sign in as Akshata again: the clinician's note is now on the result.
6. Sign in as **Site Administrator** to see the accounts and the activity log. Click **Edit** on any account to rename it, reset its password or disable it.
7. **How the models perform** (any account) shows the test results, including where the models fail.

You can also upload your own files: a brain MRI slice as PNG or JPG, or a sleep EEG recording as EDF (five minutes or longer).

## How it fits together

```
browser  (web/)  ──►  server  (app/server.py)  ──►  models  (models/*.onnx, run by ONNX Runtime)
                          │
                          └──►  database  (data/neurova.db, created on first run)
```

The models were trained in PyTorch and then exported to ONNX. That is why the app needs only ONNX Runtime, which is small, and not the whole of PyTorch, which is about 1 GB.

| Folder | What is in it |
| --- | --- |
| `app/server.py` | Every address the browser can call: accounts, sharing, analyses, admin |
| `app/mri.py` | The MRI analysis: prediction, Grad-CAM, uncertainty, stability check, unfamiliar-scan check |
| `app/sleep.py` | The sleep analysis: staging, hypnogram, comparison with the expert |
| `app/preprocess.py` | Preparing a scan or recording. Training uses this same file |
| `app/edf.py` | Reads EDF recordings |
| `app/auth.py` | Passwords, sign-in tokens and the rule for who may see which patient |
| `app/database.py` | The four tables: users, access, studies, audit |
| `app/report_pdf.py` | The downloadable one-page PDF report of a result (font in `app/fonts/`) |
| `app/demo.py` | The made-up demo accounts and their results |
| `web/` | The pages: `index.html`, `style.css`, `app.js`. No framework and no build step |
| `models/` | The trained models as ONNX files (17 MB) |
| `samples/` | 22 test scans (two of them ones the model gets wrong) and one sleep recording, none seen in training |
| `results/` | Figures and numbers from the test set, shown on the model page |
| `training/` | The PyTorch code that trained the models, and the export to ONNX |
| `tests/` | Automated checks of the server and of the pages |
| `report/` | The CDS301 report, slides, business model canvas and viva notes |

## The models

**MRI.** 7,023 images from the public Brain Tumor MRI Dataset. Before splitting, 866 duplicate images were removed, because many copies sat in both the official training and testing folders and would have inflated the score. Three models were then trained on a fresh 70/15/15 split:

| Model | Test accuracy | Changed by faint noise |
| --- | --- | --- |
| Small CNN from scratch | 89.3% | 36.5% |
| EfficientNet-B0, standard fine-tuning | 98.8% | 35.1% |
| EfficientNet-B0, fingerprint-scrub training (in use) | 98.8% | 0.6% |

The last column is the interesting one. The standard model changed its answer on a third of test scans when noise too faint to see was added, a sign that it had partly learned which source collection an image came from. Training with augmentation that destroys those fingerprints (resampling, JPEG re-compression, blur, noise) fixed it without losing accuracy.

**Sleep.** 20 nights from 10 healthy adults in Sleep-EDF Expanded, using one EEG channel (Fpz-Cz). People, not nights, were divided into training, validation and test, so the score reflects someone new. The model is weakest on N1 and sometimes calls quiet wakefulness N1.

**GRU or Transformer?** The GRU was also swapped for a Transformer encoder, with everything else the same. Its first run scored 85.3%, above the GRU's 83.5%, but with only two test people one run can be lucky, so each was trained three times with different random seeds. Over three runs the GRU averaged 84.7% (83.5 to 86.8%) and the Transformer 77.0% (66.0 to 85.3%), so the app keeps the GRU. `python training/compare_sleep.py` repeats the comparison.

**What the MRI model cannot do.** It knows four answers only, all for tumours inside the brain, and it must give one of them even for a scan of something else. Shown an eye-socket tumour found online, it answered meningioma or no tumour. So the app measures how far each scan sits from the training scans and warns when one is unfamiliar. That check flags 8 of the model's 11 test mistakes and 3.5% of the scans it gets right. It is a warning, not a guarantee.

Neither model has been tested on data from a hospital outside these public datasets.

## Retrain the models

Not needed to run the app. Training uses PyTorch and runs on a CPU.

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

## Put it online

The app needs a server, so GitHub Pages cannot host it. Hugging Face Spaces can, for free:

1. Create a Space at huggingface.co/new-space and choose **Docker** as the SDK.
2. Add these four lines at the very top of this `README.md`, so the Space knows how to start:
   ```
   ---
   title: Neurova
   sdk: docker
   app_port: 7860
   ---
   ```
3. Upload this folder's contents to the Space (leave out `.venv`, `data` and `datasets`). The `Dockerfile` here does the rest.

A free Space forgets its database when it restarts, and starts again with the demo accounts.

## Good to know

- **Data sources.** Brain Tumor MRI Dataset, M. Nickparvar, 2021 (CC BY 4.0). Sleep-EDF Expanded, B. Kemp et al., 2000, via PhysioNet (Open Data Commons Attribution). EfficientNet-B0 ImageNet weights from EfficientNet-PyTorch (Apache 2.0).
- **MySQL instead of SQLite.** `pip install pymysql cryptography`, create an empty database, and set `DATABASE_URL=mysql+pymysql://user:password@localhost/neurova` before starting.
- **Rename the demo people.** Change the names in `app/demo.py`, delete the `data` folder, and start again. Keep the emails as they are.
- **Start fresh.** Delete the `data` folder. The demo accounts come back on the next start.
- **Privacy.** The uploaded file itself is not kept. What is stored is the result, which for an MRI includes the cropped picture and its heatmap. Passwords are stored as bcrypt hashes.
