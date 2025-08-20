# MOF Metal Salt Prediction with GNNs

Here you will find the code to predict the **metal salts of a given MOF** using different types of input features.  
Each Jupyter Notebook (`Ex1.ipynb`, `Ex2.ipynb`, …) runs experiments with different combinations of descriptors.

---

## Folder Contents

| File name  | Scherrer | Microstrain | OMS | Atomic No. |
|------------|----------|-------------|-----|------------|
| **Ex1.ipynb**  | ✗ | ✗ | ✗ | ✗ |
| **Ex2.ipynb**  | ✓ | ✓ | ✓ | ✗ |
| **Ex3.ipynb**  | ✓ | ✓ | ✗ | ✗ |
| **Ex4.ipynb**  | ✓ | ✗ | ✓ | ✗ |
| **Ex5.ipynb**  | ✗ | ✗ | ✓ | ✗ |
| **Ex6.ipynb**  | ✗ | ✗ | ✗ | ✓ |
| **Ex7.ipynb**  | ✗ | ✗ | ✓ | ✓ |
| **Ex8.ipynb**  | ✓ | ✗ | ✓ | ✓ |
| **Ex9.ipynb**  | ✓ | ✓ | ✓ | ✓ |
| **Ex10.ipynb** | ✓ | ✓ | ✗ | ✓ |



Each notebook combines these features differently, as shown in the table above.

---

## Notebook Structure

Every notebook (`Ex*.ipynb`) follows the same workflow:

1. **Load the Data**  
   Import and prepare the dataset for training and evaluation.

2. **Define the GNN Model**  
   Specify the Graph Neural Network architecture (layers, activation functions, etc.).

3. **Train the Model**  
   - Train the model on the selected features.  
   - Store the trained weights in the `tmp/` folder.  

   ⚠️ **Note**:  
   If you only want to **test the model** without re-training, you can **skip this section** and directly load the pre-trained weights.

4. **Load and Evaluate the Model**  
   - Load the saved model weights.  
   - Evaluate the model performance on the dataset.

---

## Usage

- Pick the notebook (`Ex*.ipynb`) corresponding to the feature set you want to test.  
- Run all cells to train and evaluate the GNN, **or skip section 3 if you only want to test using pre-trained models**.  
