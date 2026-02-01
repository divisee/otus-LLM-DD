#!/usr/bin/env python3
"""
Скрипт для генерации диаграммы работы агентов LangGraph.
Заменяет langgraph.png в report/screenshots/
"""

import graphviz
import os

# Создаем директорию, если не существует
output_dir = os.path.join(os.path.dirname(__file__), '..', 'report', 'screenshots')
os.makedirs(output_dir, exist_ok=True)

# Создаем диаграмму
dot = graphviz.Digraph(
    comment='Movie Agent LangGraph',
    graph_attr={'rankdir': 'LR', 'fontsize': '12'},
    node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': 'lightblue', 'fontsize': '10'},
    edge_attr={'fontsize': '10'}
)

# Узлы агентов
dot.node('analyzer', 'AnalyzerAgent\nАнализирует запрос\nОпределяет need_rag/need_search')
dot.node('gatherer', 'GatherAgent\nСобирает данные\nRAG + Web Search')
dot.node('answerer', 'AnswerAgent\nФормирует ответ\nНа основе собранных данных')
dot.node('reviewer', 'ReviewAgent\nПроверяет качество\nМожет вернуть на Gather')

# Ребра
dot.edge('analyzer', 'gatherer', label='JSON с флагами')
dot.edge('gatherer', 'answerer', label='Собранные данные')
dot.edge('answerer', 'reviewer', label='JSON с ответом')
dot.edge('reviewer', 'gatherer', label='refine_needed\n(max 3 итерации)', style='dashed', color='red')

# Сохраняем как PNG
output_path = os.path.join(output_dir, 'langgraph')
dot.render(output_path, format='png', cleanup=True)

print(f"Диаграмма сохранена в {output_path}.png")
