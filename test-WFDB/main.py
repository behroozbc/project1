import wfdb
from scipy.io import wavfile
import numpy as np

# خواندن رکورد WFDB (مثال: 'mitdb/100')
# فایل ورودی باید بدون پسوند باشه
record = wfdb.rdrecord("E:/Work/University/PR/datas/voice-icar-federico-ii-database-1.0.0/voice001")  # مسیر فایل را جایگزین کنید

# سیگنال (معمولاً کانال اول برای تک‌کاناله)
signal = record.p_signal[:, 0]  # اگر چند کاناله، ایندکس کانال را انتخاب کنید

# نرخ نمونه‌برداری (fs)
fs = record.fs  # مثلاً 360 Hz برای ECG

# نرمال‌سازی سیگنال به محدوده صوتی (-1 تا 1 برای float، یا int16)
# برای WAV، بهتر است به int16 تبدیل شود (محدوده -32768 تا 32767)
signal_normalized = np.int16(signal / np.max(np.abs(signal)) * 32767)

# ذخیره به عنوان WAV (مونو، نرخ fs)
wavfile.write('output.wav', fs, signal_normalized)

print("فایل WAV با موفقیت ذخیره شد!")