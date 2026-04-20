from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# ─── Load & preprocess ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'train.csv')

df = pd.read_csv(CSV_PATH)

# Drop Cabin (too many nulls)
df = df.drop(columns=['Cabin'])

# Fill missing values (pandas 2.x compatible - no inplace on chained ops)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categoricals
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Features & target
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[feature_cols]
Y = df['Survived']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

train_acc = round(accuracy_score(Y_train, model.predict(X_train)) * 100, 2)
test_acc  = round(accuracy_score(Y_test,  model.predict(X_test))  * 100, 2)

print(f"✅ Model trained — Train: {train_acc}%  |  Test: {test_acc}%")

# ─── EDA stats for the UI ─────────────────────────────────────────────────────
total              = len(df)
survived_count     = int(df['Survived'].sum())
not_survived_count = total - survived_count

sex_grp    = df.groupby('Sex')['Survived'].agg(['sum', 'count'])
pclass_grp = df.groupby('Pclass')['Survived'].agg(['sum', 'count'])

def safe_get(grp, key, col):
    return int(grp.loc[key, col]) if key in grp.index else 0

eda_stats = {
    'total':         total,
    'survived':      survived_count,
    'not_survived':  not_survived_count,
    'survival_rate': round(survived_count / total * 100, 1),
    'train_acc':     train_acc,
    'test_acc':      test_acc,
    # Gender (0=male, 1=female after encoding)
    'sex_survived': [safe_get(sex_grp, 0, 'sum'), safe_get(sex_grp, 1, 'sum')],
    'sex_total':    [safe_get(sex_grp, 0, 'count'), safe_get(sex_grp, 1, 'count')],
    # Pclass
    'pclass_labels':   ['1st Class', '2nd Class', '3rd Class'],
    'pclass_survived': [safe_get(pclass_grp, i, 'sum')   for i in [1, 2, 3]],
    'pclass_total':    [safe_get(pclass_grp, i, 'count') for i in [1, 2, 3]],
    # Misc
    'age_mean':  round(float(df['Age'].mean()), 1),
    'fare_mean': round(float(df['Fare'].mean()), 1),
}

# Age distribution
age_bins   = [0, 10, 20, 30, 40, 50, 60, 80]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+']
eda_stats['age_dist']   = [
    int(((df['Age'] >= age_bins[i]) & (df['Age'] < age_bins[i+1])).sum())
    for i in range(len(age_bins) - 1)
]
eda_stats['age_labels'] = age_labels

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', stats=eda_stats)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            int(data['pclass']),
            0 if data['sex'] == 'male' else 1,
            float(data['age']),
            int(data['sibsp']),
            int(data['parch']),
            float(data['fare']),
            int(data['embarked']),
        ]])
        pred      = int(model.predict(features)[0])
        proba     = model.predict_proba(features)[0]
        return jsonify({
            'survived':        pred,
            'survive_prob':    round(float(proba[1]) * 100, 1),
            'not_survive_prob':round(float(proba[0]) * 100, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
