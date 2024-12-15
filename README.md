# TypeDA

Data and code for the CoLING 2025 paper "Large Language Models are Good Annotators for Type-aware Data Augmentation in Grammatical Error Correction"

**Code Structure**

```latex
TypeDA                            # Root
└── data                          # Data for augmentation and correction(json, m2)
└── checkpoint                    # Trained model chekpoints and pre-trained model
└── evaluation                    # Evaluation Method
    └── augmenters                # Tools to calculate affinity and diversity
    └── m2scorer                  # Calculate precision, recall and F0.5 score
    └── metrics                   # Calculate affinity and diversity
└── module                        # Main method
    └── augmentation              # Method for TypeDA
        └── filling				  # Fill the masked data with GPT-4
        	└── fill              # Main method for error filling
        	└── postprocess       # Postprocess method
        	└── template          # Prompting template construction
        	└── tool              # Other tools for filling
        └── mask                  # Masking the original data 
            └── dataset           # Data preparation
        	└── infer             # Infer to get the masked data
        	└── method            # Multi-decoder model and customized loss function
        	└── model             # Multi-decoder model and customized loss function
        	└── trainer			  # Train the mask model
        └── config			  	  # Config settings
    └── correction                # Method for grammatical error correction
└── module                        # Other tools

```

To start with, you can input the following commands into the terminal to configure the environment.

```
pip install -r requirements.txt
```

Please place the M2-format data used for augmentation into the **'data/m2/'** directory, place the pre-trained model files into the **'checkpoint/'** directory, and specify the GPT API key in **'module/config.py'**.

Then you can simply run:

```
python main.py
```

For the grammar correction model, the default backbone model is 'google-t5/t5-base'. You can download other pre-trained language models and modify the corresponding model name in 'module/correction/gec.py'. For the GECToR model, please refer to https://github.com/grammarly/gector.

The code of M2scorer is adapted from https://github.com/nusnlp/m2scorer

The code for calculating affinity and diversity is adapted from https://github.com/THUKElab/MixEdit/tree/main

