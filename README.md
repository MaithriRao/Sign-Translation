# AVASAG: Avatar-Based Language Assistant for Automated Sign Translation

## Project Description

This project focuses on developing an NLP system for automated translation between German text and sign language glosses. It aims to bridge the communication gap between hearing and deaf communities by providing accurate and efficient translation.

## Key Features

* **Bidirectional Translation:** Translates German text to sign language glosses and vice versa.
* **Low-Resource Dataset:** Utilizes a limited dataset, addressing the challenge of data scarcity in sign language translation.
* **Data Augmentation:** Employs data augmentation techniques to enhance the dataset and improve model training.
* **Seq2Seq Transformer Model:** Implements a Seq2Seq Transformer model for effective translation.
* **SentencePiece Tokenizer:** Uses the SentencePiece tokenizer to handle vocabulary and subword units.
* **NLLB Model Integration:** Leverages the NLLB model for comparison and performance enhancement.
* **LLaMA Fine-tuning:** Fine-tunes the LLaMA3.1 model to improve translation efficacy.

## Results

* **Dataset Augmentation Impact:** Achieved a 64.83% improvement in translation quality through dataset augmentation.
* **NLLB Performance:** Attained BLEU score enhancements of 36.9% and 65.3% when comparing original vs. augmented data using the NLLB model.
* **LLaMA Fine-tuning Success:** Achieved BLEU score enhancements of 36.9% (Text-to-Gloss) and 65.3% (Gloss-to-Text) by fine-tuning LLaMA3.1 on the entire dataset.

## Technologies Used

* Python
* Seq2Seq Transformer
* SentencePiece
* NLLB Model
* LLaMA3.1

## Usage

(If applicable, add instructions on how to run or use the project. If not, you can omit this section.)

## Contributing

(If applicable, add information on how others can contribute to the project. If not, you can omit this section.)

## License

(If applicable, add license information. If not, you can omit this section.)

## Contact

(Optional: Add your contact information.)
