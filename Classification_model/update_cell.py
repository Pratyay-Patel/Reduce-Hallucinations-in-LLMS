import json

notebook_path = 'train_advanced.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find the cell we added previously and replace it
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "def evaluate_with_threshold" in source_str:
            new_source = [
                "def evaluate_with_threshold(model, X, y, threshold=0.5, positive_class=1):\n",
                "    from sklearn.metrics import accuracy_score, f1_score\n",
                "    import numpy as np\n",
                "    \n",
                "    if not hasattr(model, 'predict_proba'):\n",
                "        return None, None\n",
                "        \n",
                "    probs = model.predict_proba(X)\n",
                "    y_pred = (probs[:, positive_class] >= threshold).astype(int)\n",
                "    acc = accuracy_score(y, y_pred)\n",
                "    f1 = f1_score(y, y_pred, average='macro')\n",
                "    return acc, f1\n",
                "\n",
                "def find_best_threshold(model, X, y, positive_class=1):\n",
                "    import numpy as np\n",
                "    \n",
                "    thresholds = np.arange(0.1, 0.95, 0.05)\n",
                "    best_acc = 0\n",
                "    best_thresh_acc = 0\n",
                "    best_f1 = 0\n",
                "    best_thresh_f1 = 0\n",
                "    \n",
                "    print(\"Evaluating Thresholds...\")\n",
                "    for t in thresholds:\n",
                "        acc, f1 = evaluate_with_threshold(model, X, y, threshold=t, positive_class=positive_class)\n",
                "        if acc is None: return\n",
                "        \n",
                "        if acc > best_acc:\n",
                "            best_acc = acc\n",
                "            best_thresh_acc = t\n",
                "            \n",
                "        if f1 > best_f1:\n",
                "            best_f1 = f1\n",
                "            best_thresh_f1 = t\n",
                "            \n",
                "    print(f\"Best Threshold by Accuracy: {best_thresh_acc:.2f} (Acc: {best_acc:.4f})\")\n",
                "    print(f\"Best Threshold by F1-Macro: {best_thresh_f1:.2f} (F1: {best_f1:.4f})\")\n",
                "\n",
                "    return best_thresh_acc, best_thresh_f1\n",
                "\n",
                "# Example usage (uncomment to run with your best model):\n",
                "# best_t_acc, best_t_f1 = find_best_threshold(best_model_obj, X_test_final, y_test, positive_class=1)\n"
            ]
            nb['cells'][idx]['source'] = new_source
            break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Updated notebook to loop through thresholds!")
