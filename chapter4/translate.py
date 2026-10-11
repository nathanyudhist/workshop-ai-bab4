import os
import re
import tokenize
import tokenize as tk
import io

def generate_translations():
    identifier_map = {
        'jumlah_sampel': 'num_samples',
        'panjang_urutan': 'seq_length',
        'jumlah_fitur': 'num_features',
        'laju_belajar': 'learning_rate',
        'galat': 'error',
        'gradien': 'gradient',
        'kamera': 'camera',
        'ukuran': 'size',
        'NAMA_MODEL': 'MODEL_NAME',
        'PILIHAN': 'MODEL_OPTIONS',
        'konteks': 'context',
        'pertanyaan': 'questions',
        'penjawab': 'qa_pipeline',
        'kalimat': 'sentences',
        'teks': 'text',
        'hasil': 'result',
        'riwayat': 'history',
        'kode': 'code',
        'rekonstruksi': 'reconstruction',
        'LANGKAH': 'STEPS',
        'buat_jendela': 'make_windows',
        'tampilkan': 'show',
        'evaluasi': 'evaluate',
        'akurasi': 'accuracy',
        'waktu': 'time',
        'lapisan': 'layer',
        'detik': 'seconds',
        'soal': 'task',
        'bobot': 'weights',
        'awal': 'initial',
        'dengan': 'with',
        'untuk': 'for',
        'yang': 'that',
        'tidak': 'not',
        'atau': 'or',
        'gambar': 'image',
        'contoh': 'example',
        'contoh_teks': 'example_text',
        'jumlah': 'num',
        'label_uji': 'test_labels',
        'label_latih': 'train_labels',
        'data_uji': 'test_data',
        'data_latih': 'train_data',
        'jumlah_kelas': 'num_classes',
        'jumlah_epoch': 'num_epochs',
        'model_awal': 'initial_model',
        'model_baru': 'new_model',
        'selisih': 'difference',
        'waktu_training': 'training_time',
        'akurasi_test': 'test_accuracy',
        'akurasi_uji': 'test_accuracy',
        'prediksi': 'prediction',
        'kelas': 'class',
        'kategori': 'category'
    }

    # Order matters for phrases!
    phrases_map = {
        'Total parameter yang diharapkan': 'Expected total parameters',
        'Total parameter:': 'Total parameters:',
        'Total parameter': 'Total parameters',
        'Waktu training (detik)': 'Training time (seconds)',
        'waktu training': 'training time',
        'Akurasi test': 'Test accuracy',
        'Akurasi': 'Accuracy',
        'akurasi': 'accuracy',
        'File gambar': 'Image file',
        'tidak ditemukan': 'not found',
        'Soal:': 'Task:',
        'Cara pakai:': 'Usage:',
        'Catatan:': 'Note:',
        'Petunjuk:': 'Hint:',
        'Contoh': 'Example',
        'contoh': 'example',
        'KOMENTAR HASIL:': 'RESULT COMMENTS:',
        'Arsitektur mirip': 'Architecture similar to',
        'Hanya membuat model dan menampilkan ringkasannya': 'Only creates the model and prints its summary',
        'Model awal': 'Initial model',
        'Model dengan blok tambahan': 'Model with additional block',
        'Model baru': 'New model',
        'Parameter model awal': 'Initial model parameters',
        'Parameter model baru': 'New model parameters',
        'Selisih': 'Difference',
        'Filter C1': 'C1 Filters',
        'Filter C2': 'C2 Filters',
        'Neuron': 'Neurons',
        'Waktu training': 'Training time',
        'detik': 'seconds',
        'lapisan': 'layer',
        'awal': 'initial',
        'dengan': 'with',
        'untuk': 'for',
        'yang': 'that',
        'tidak': 'not',
        'atau': 'or',
        'hasil': 'result',
        'gambar': 'image',
        'buat_jendela': 'make_windows',
        'tampilkan': 'show',
        'evaluasi': 'evaluate',
        'kode': 'code',
        'kalimat': 'sentences',
        'teks': 'text',
        'pertanyaan': 'questions',
        'konteks': 'context',
        'PILIHAN': 'OPTIONS',
        'NAMA_MODEL': 'MODEL_NAME',
        'kamera': 'camera',
        'ukuran': 'size',
        'gradien': 'gradient',
        'galat': 'error',
        'laju_belajar': 'learning_rate',
        'jumlah_fitur': 'num_features',
        'panjang_urutan': 'seq_length',
        'jumlah_sampel': 'num_samples',
        'waktu': 'time',
        'soal': 'task',
        'bobot': 'weights',
        'jumlah': 'num',
        'jumlah_kelas': 'num_classes',
        'jumlah_epoch': 'num_epochs'
    }

    # Separate words that need word boundary regex
    word_translations = {
        'dengan': 'with',
        'untuk': 'for',
        'yang': 'that',
        'tidak': 'not',
        'atau': 'or',
        'contoh': 'example',
        'awal': 'initial',
        'tampil': 'show',
        'hasil': 'result',
        'gambar': 'image',
        'kamera': 'camera',
        'ukuran': 'size',
        'jumlah': 'number',
        'akurasi': 'accuracy',
        'waktu': 'time',
        'lapisan': 'layer',
        'detik': 'seconds',
        'soal': 'task',
        'bobot': 'weights',
        'kode': 'code',
        'kalimat': 'sentence',
        'pertanyaan': 'question',
        'konteks': 'context',
        'galat': 'error',
        'gradien': 'gradient',
        'teks': 'text'
    }

    return identifier_map, phrases_map, word_translations

identifier_map, phrases_map, word_translations = generate_translations()

word_regexes = [(re.compile(r'\b' + k + r'\b', re.IGNORECASE), v) for k, v in word_translations.items()]

def translate_string(s):
    # First apply phrases map
    for k, v in phrases_map.items():
        # Using regex to replace case-insensitively but maintaining the case of the replacement as is.
        # It's safer to just do a direct replace for case-sensitive phrases
        s = s.replace(k, v)
    
    # Then apply word regexes
    for regex, v in word_regexes:
        s = regex.sub(v, s)
    
    return s

def process_file(filepath):
    if filepath.endswith('translate.py') or filepath.endswith('extract.py'):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tokens = list(tk.generate_tokens(io.StringIO(source).readline))
    except tk.TokenError:
        print(f"TokenError in {filepath}")
        return

    out_tokens = []
    changed = False

    for tok in tokens:
        tok_type, tok_string, start, end, line = tok
        new_string = tok_string

        if tok_type == tk.NAME:
            # Identifier
            if tok_string in identifier_map:
                new_string = identifier_map[tok_string]
                changed = True
        
        elif tok_type == tk.STRING:
            # Could be f-string or normal string
            # It's tricky to translate contents without messing up python quotes
            # Let's extract the inside of the string
            prefix = ''
            quote = ''
            content = tok_string
            m = re.match(r'^([rRuUbBfF]*)(["\']+)', tok_string)
            if m:
                prefix = m.group(1)
                quote = m.group(2)
                content = tok_string[len(prefix) + len(quote): -len(quote)]
                
                # We should NOT translate English library contents. 
                # Let's just do a naive translation on the content.
                new_content = translate_string(content)
                if new_content != content:
                    new_string = f"{prefix}{quote}{new_content}{quote}"
                    changed = True

        elif tok_type == tk.COMMENT:
            # Comment
            content = tok_string[1:]
            new_content = translate_string(content)
            if new_content != content:
                new_string = f"#{new_content}"
                changed = True
        
        out_tokens.append((tok_type, new_string))

    if changed:
        new_source = tk.untokenize(out_tokens)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_source)
        print(f"Translated: {filepath}")

for root, dirs, filenames in os.walk('.'):
    for f in filenames:
        if f.endswith('.py'):
            process_file(os.path.join(root, f))

print("Translation script finished.")
