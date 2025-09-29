"""
Análisis de Regresión para Precios de Viviendas
Dataset: House Prices - Advanced Regression Techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Carga y prepara los datos para el análisis"""
    try:
        df = pd.read_csv('train.csv')
        print("✅ Datos cargados exitosamente")
        print(f"📊 Dimensiones del dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo train.csv")
        print("📥 Por favor descárgalo de: https://www.kaggle.com/c/house-prices-advanced-regression-techniques")
        return None

def preprocess_data(df):
    """Preprocesa los datos seleccionando características y limpiando valores nulos"""
    # Seleccionar características iniciales
    features = ['GrLivArea', 'YearBuilt']
    target = 'SalePrice'
    
    # Verificar que las columnas existen
    for col in features + [target]:
        if col not in df.columns:
            print(f"❌ Error: La columna {col} no existe en el dataset")
            return None, None, None, None, None, None
    
    # Limpiar datos
    df_clean = df[features + [target]].dropna()
    print(f"🧹 Datos después de la limpieza: {df_clean.shape}")
    
    # Separar características y variable objetivo
    X = df_clean[features]
    y = df_clean[target]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📚 Conjunto de entrenamiento: {X_train.shape}")
    print(f"🧪 Conjunto de prueba: {X_test.shape}")
    
    # Estandarizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Entrena y evalúa múltiples modelos de regresión"""
    # Definir los modelos
    models = {
        'Linear Regression': LinearRegression(),
        'SVM': SVR(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("🚀 ENTRENAMIENTO DE MODELOS")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n📊 Entrenando {name}...")
        
        # Usar datos escalados para SVM, sin escalar para otros modelos
        if name == 'SVM':
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_te = X_test
        
        # Entrenar el modelo
        model.fit(X_tr, y_train)
        
        # Predecir
        y_pred = model.predict(X_te)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Almacenar resultados
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
        
        print(f"   ✅ {name} - MSE: {mse:,.2f}, RMSE: {np.sqrt(mse):,.2f}, R²: {r2:.4f}")
    
    return results

def compare_models(results):
    """Compara los resultados de todos los modelos"""
    # Crear tabla de comparación
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'MSE': [results[name]['mse'] for name in results.keys()],
        'RMSE': [results[name]['rmse'] for name in results.keys()],
        'R²': [results[name]['r2'] for name in results.keys()]
    }).sort_values('MSE')
    
    print("\n" + "="*60)
    print("📈 COMPARACIÓN DE MODELOS")
    print("="*60)
    print(comparison_df.round(4))
    
    # Encontrar el mejor modelo
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_mse = comparison_df.iloc[0]['MSE']
    best_model_rmse = comparison_df.iloc[0]['RMSE']
    best_model_r2 = comparison_df.iloc[0]['R²']
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   📉 MSE: {best_model_mse:,.2f}")
    print(f"   📏 RMSE: {best_model_rmse:,.2f}")
    print(f"   📊 R²: {best_model_r2:.4f}")
    
    return comparison_df

def visualize_results(results, X_test, y_test):
    """Crea visualizaciones de los resultados"""
    # Configurar el estilo de las gráficas
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Colores para diferentes modelos
    colors = ['blue', 'red', 'green', 'orange']
    
    # Gráfico para cada modelo
    for idx, (name, color) in enumerate(zip(results.keys(), colors)):
        y_pred = results[name]['predictions']
        
        # Gráfico de dispersión: Valores reales vs predichos
        axes[idx].scatter(y_test, y_pred, alpha=0.6, color=color, s=50)
        
        # Línea de perfecta predicción
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2)
        
        axes[idx].set_xlabel('Valores Reales (SalePrice)')
        axes[idx].set_ylabel('Valores Predichos')
        axes[idx].set_title(f'{name}\nMSE: {results[name]["mse"]:,.2f}, R²: {results[name]["r2"]:.4f}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfica de comparación de errores
    plt.figure(figsize=(10, 6))
    models_names = list(results.keys())
    mse_values = [results[name]['mse'] for name in models_names]
    
    bars = plt.bar(models_names, mse_values, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparación de MSE entre Modelos')
    plt.xticks(rotation=45)
    
    # Añadir valores en las barras
    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01, 
                f'{value:,.0f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def advanced_analysis(df):
    """Análisis avanzado con más características"""
    print("\n" + "="*60)
    print("🔍 ANÁLISIS AVANZADO CON MÁS CARACTERÍSTICAS")
    print("="*60)
    
    # Seleccionar más características relevantes
    advanced_features = [
        'GrLivArea', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea'
    ]
    target = 'SalePrice'
    
    # Filtrar características que existen en el dataset
    available_features = [f for f in advanced_features if f in df.columns]
    print(f"Características seleccionadas: {available_features}")
    
    # Limpiar datos
    df_advanced = df[available_features + [target]].dropna()
    print(f"Dataset para análisis avanzado: {df_advanced.shape}")
    
    X_adv = df_advanced[available_features]
    y_adv = df_advanced[target]
    
    # Dividir datos
    X_train_adv, X_test_adv, y_train_adv, y_test_adv = train_test_split(
        X_adv, y_adv, test_size=0.2, random_state=42
    )
    
    # Entrenar Random Forest (mejor modelo del análisis simple)
    rf_advanced = RandomForestRegressor(random_state=42)
    rf_advanced.fit(X_train_adv, y_train_adv)
    
    # Predecir y evaluar
    y_pred_adv = rf_advanced.predict(X_test_adv)
    mse_adv = mean_squared_error(y_test_adv, y_pred_adv)
    r2_adv = r2_score(y_test_adv, y_pred_adv)
    
    print(f"\n📊 Random Forest con {len(available_features)} características:")
    print(f"   ✅ MSE: {mse_adv:,.2f}, RMSE: {np.sqrt(mse_adv):,.2f}, R²: {r2_adv:.4f}")
    
    # Importancia de características
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_advanced.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Importancia de características:")
    print(feature_importance.round(4))
    
    return feature_importance

def main():
    """Función principal"""
    print("🏠 ANÁLISIS DE REGRESIÓN PARA PRECIOS DE VIVIENDAS")
    print("="*60)
    
    # 1. Cargar datos
    df = load_and_prepare_data()
    if df is None:
        return
    
    # 2. Preprocesamiento básico
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    if X_train is None:
        return
    
    # 3. Entrenar y evaluar modelos
    results = train_and_evaluate_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 4. Comparar modelos
    comparison_df = compare_models(results)
    
    # 5. Visualizar resultados
    visualize_results(results, X_test, y_test)
    
    # 6. Análisis avanzado
    feature_importance = advanced_analysis(df)
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("📁 Archivos generados:")
    print("   - model_comparison.png (Comparación de predicciones)")
    print("   - mse_comparison.png (Comparación de errores)")
    print("\n💡 Conclusiones:")
    print("   - El modelo con menor MSE es el más preciso")
    print("   - R² cercano a 1 indica mejor ajuste")
    print("   - Random Forest suele funcionar bien con datos estructurados")

if __name__ == "__main__":
    main()
