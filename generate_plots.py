#!/usr/bin/env python3
"""
Script para generar gráficas del Lab 9 - Chat-Box CUDA
Requisitos: pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def plot_stream_comparison():
    """Gráfica principal: Comparación de rendimiento por streams"""
    df = pd.read_csv('benchmark_results.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de Rendimiento: 1, 2, 4, 8 Streams', 
                 fontsize=16, fontweight='bold')
    
    # 1. Latencia p50 y p95
    ax1 = axes[0, 0]
    x = np.arange(len(df['num_streams']))
    width = 0.35
    ax1.bar(x - width/2, df['p50_ms'], width, label='p50', color='#2ecc71', alpha=0.8)
    ax1.bar(x + width/2, df['p95_ms'], width, label='p95', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Número de Streams')
    ax1.set_ylabel('Latencia (ms)')
    ax1.set_title('Latencia p50 y p95 por Configuración')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['num_streams'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. QPS (Queries Per Second)
    ax2 = axes[0, 1]
    ax2.plot(df['num_streams'], df['qps'], marker='o', linewidth=2.5, 
             markersize=10, color='#3498db')
    ax2.fill_between(df['num_streams'], 0, df['qps'], alpha=0.3, color='#3498db')
    ax2.set_xlabel('Número de Streams')
    ax2.set_ylabel('QPS (queries/segundo)')
    ax2.set_title('Throughput por Configuración')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(df['num_streams'])
    
    # 3. Desglose por etapa (stacked bar)
    ax3 = axes[1, 0]
    bottom = np.zeros(len(df))
    colors = ['#9b59b6', '#f39c12', '#1abc9c']
    stages = ['avg_nlu_ms', 'avg_data_ms', 'avg_fuse_ms']
    labels = ['NLU', 'DATA', 'FUSE']
    
    for stage, label, color in zip(stages, labels, colors):
        ax3.bar(df['num_streams'], df[stage], bottom=bottom, 
                label=label, color=color, alpha=0.8)
        bottom += df[stage]
    
    ax3.set_xlabel('Número de Streams')
    ax3.set_ylabel('Tiempo Promedio (ms)')
    ax3.set_title('Desglose de Latencia por Etapa')
    ax3.legend()
    ax3.set_xticks(df['num_streams'])
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Speedup relativo (vs 1 stream)
    ax4 = axes[1, 1]
    baseline_qps = df[df['num_streams'] == 1]['qps'].values[0]
    speedup = df['qps'] / baseline_qps
    
    ax4.bar(df['num_streams'], speedup, color='#e67e22', alpha=0.8)
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Baseline (1 stream)')
    ax4.set_xlabel('Número de Streams')
    ax4.set_ylabel('Speedup (×)')
    ax4.set_title('Aceleración Relativa (vs 1 Stream)')
    ax4.set_xticks(df['num_streams'])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Anotaciones de valores
    for i, (streams, sp) in enumerate(zip(df['num_streams'], speedup)):
        ax4.text(streams, sp + 0.05, f'{sp:.2f}×', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('stream_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica guardada: stream_comparison.png")
    plt.show()

def plot_detailed_metrics():
    """Gráficas detalladas de métricas por consulta"""
    df = pd.read_csv('metrics.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análisis Detallado por Consulta', fontsize=16, fontweight='bold')
    
    # 1. Latencia total por query
    ax1 = axes[0, 0]
    ax1.plot(df['query_id'], df['lat_total_ms'], marker='o', linewidth=1.5, 
             markersize=6, color='#3498db', label='Latencia Total')
    ax1.axhline(y=df['lat_total_ms'].mean(), color='red', linestyle='--', 
                label=f'Promedio: {df["lat_total_ms"].mean():.2f}ms')
    ax1.set_xlabel('Query ID')
    ax1.set_ylabel('Latencia Total (ms)')
    ax1.set_title('Latencia por Consulta')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución de intenciones
    ax2 = axes[0, 1]
    intent_counts = df['top_intent'].value_counts()
    colors_pie = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#2ecc71']
    ax2.pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 9})
    ax2.set_title('Distribución de Intenciones Detectadas')
    
    # 3. Comparación de etapas
    ax3 = axes[1, 0]
    stages_df = df[['lat_nlu_ms', 'lat_data_ms', 'lat_fuse_ms']].mean()
    bars = ax3.barh(['NLU', 'DATA', 'FUSE'], stages_df.values, 
                     color=['#9b59b6', '#f39c12', '#1abc9c'], alpha=0.8)
    ax3.set_xlabel('Tiempo Promedio (ms)')
    ax3.set_title('Tiempo Promedio por Etapa del Pipeline')
    ax3.grid(axis='x', alpha=0.3)
    
    # Anotaciones
    for bar, val in zip(bars, stages_df.values):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.3f}ms', va='center', fontweight='bold')
    
    # 4. Heatmap de sensores promedio
    ax4 = axes[1, 1]
    sensor_cols = ['mean_mov', 'mean_lux', 'mean_temp', 'mean_ruido', 'mean_hum']
    sensor_data = df[sensor_cols].mean().values.reshape(1, -1)
    sensor_labels = ['Movimiento', 'Luz (lux)', 'Temp (°C)', 'Ruido (dB)', 'Humedad']
    
    im = ax4.imshow(sensor_data, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(sensor_labels)))
    ax4.set_xticklabels(sensor_labels, rotation=45, ha='right')
    ax4.set_yticks([0])
    ax4.set_yticklabels(['Promedio'])
    ax4.set_title('Valores Promedio de Sensores')
    
    # Anotaciones de valores
    for i, (label, val) in enumerate(zip(sensor_labels, sensor_data[0])):
        ax4.text(i, 0, f'{val:.1f}', ha='center', va='center', 
                 color='white', fontweight='bold', fontsize=10)
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('detailed_metrics.png', dpi=300, bbox_inches='tight')
    print(" Gráfica guardada: detailed_metrics.png")
    plt.show()

def plot_stage_breakdown():
    """Gráfica de cascada para mostrar el pipeline"""
    df = pd.read_csv('benchmark_results.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stages = ['avg_nlu_ms', 'avg_data_ms', 'avg_fuse_ms']
    labels = ['NLU', 'DATA', 'FUSE']
    colors = ['#9b59b6', '#f39c12', '#1abc9c']
    
    x = df['num_streams']
    bottom = np.zeros(len(df))
    
    for stage, label, color in zip(stages, labels, colors):
        ax.plot(x, bottom + df[stage]/2, 'o-', linewidth=2, 
                markersize=8, label=label, color=color)
        ax.fill_between(x, bottom, bottom + df[stage], alpha=0.3, color=color)
        bottom += df[stage]
    
    ax.set_xlabel('Número de Streams', fontsize=12)
    ax.set_ylabel('Tiempo Acumulado (ms)', fontsize=12)
    ax.set_title('Pipeline de Procesamiento: Desglose por Etapa', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig('stage_breakdown.png', dpi=300, bbox_inches='tight')
    print(" Gráfica guardada: stage_breakdown.png")
    plt.show()

def generate_summary_table():
    """Genera tabla resumen en formato LaTeX para el informe"""
    df = pd.read_csv('benchmark_results.csv')
    
    print("\n" + "="*60)
    print("TABLA RESUMEN PARA INFORME (formato LaTeX)")
    print("="*60)
    
    print("\n\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparación de Rendimiento por Configuración de Streams}")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Streams} & \\textbf{p50 (ms)} & \\textbf{p95 (ms)} & " +
          "\\textbf{QPS} & \\textbf{NLU (ms)} & \\textbf{DATA (ms)} & \\textbf{FUSE (ms)} \\\\")
    print("\\hline")
    
    for _, row in df.iterrows():
        print(f"{int(row['num_streams'])} & {row['p50_ms']:.3f} & {row['p95_ms']:.3f} & " +
              f"{row['qps']:.2f} & {row['avg_nlu_ms']:.3f} & " +
              f"{row['avg_data_ms']:.3f} & {row['avg_fuse_ms']:.3f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}\n")
    
    print("="*60)
    print("TABLA RESUMEN (formato Markdown)")
    print("="*60 + "\n")
    print(df.to_markdown(index=False, floatfmt=".3f"))
    print()

def main():
    """Función principal"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     GENERADOR DE GRÁFICAS - LAB 9 CHAT-BOX CUDA          ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    try:
        # Verificar archivos
        print("Verificando archivos CSV...")
        pd.read_csv('benchmark_results.csv')
        pd.read_csv('metrics.csv')
        print(" Archivos encontrados\n")
        
        # Generar gráficas
        print("Generando gráficas...\n")
        plot_stream_comparison()
        plot_detailed_metrics()
        plot_stage_breakdown()
        
        # Generar tabla resumen
        generate_summary_table()
        
        print("\n╔═══════════════════════════════════════════════════════════╗")
        print("║   TODAS LAS GRÁFICAS SE GENERARON EXITOSAMENTE             ║")
        print("╚═══════════════════════════════════════════════════════════╝")
        print("\nArchivos generados:")
        print("  1. stream_comparison.png    - Comparación principal")
        print("  2. detailed_metrics.png     - Análisis detallado")
        print("  3. stage_breakdown.png      - Pipeline por etapa")
        print("\nUsa estas imágenes en tu informe técnico.")
        
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}")
        print("   Asegúrate de haber ejecutado primero: ./chatbox_cuda")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()