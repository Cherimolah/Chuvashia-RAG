import json
import re
from collections import defaultdict

# Исключаемые URL
excluded = {
    "https://chuvash.org/news/24.html",
    "https://chuvash.org/news/28.html", 
    "https://chuvash.org/news/16.html",
    "https://chuvash.org/news/9.html",
    "https://chuvash.org/news/10.html",
    "https://chuvash.org/news/60.html",
    "https://chuvash.org/news/13.html",
}

def has_proper_nouns(text):
    """Проверяет наличие имён собственных"""
    caps = len(re.findall(r'\b[А-ЯЁ][а-яёӑҫӗ]*(?:\s+[А-ЯЁ][а-яёӑҫӗ]*)*\b', text))
    return caps >= 2

def has_numbers(text):
    """Проверяет наличие чисел"""
    nums = re.findall(r'\d+', text)
    return len(nums) > 0

def score_article(art):
    """Оценивает статью"""
    text = art.get('text', '')
    title = art.get('title', '')
    
    if len(text) < 100:
        return -1
    
    score = 0
    if has_proper_nouns(text + title):
        score += 3
    if has_numbers(text):
        score += 2
    if len(text) > 300:
        score += 1
    
    return score

def categorize(text, title):
    """Определяет категории"""
    full = (text + title).lower()
    cats = []
    
    # Спорт
    if any(w in full for w in ['спорт', 'чемпион', 'медал', 'атлетик', 'рекорд', 'побед', 'турнир', 'гимнаст']):
        cats.append('sport')
    
    # Культура
    if any(w in full for w in ['театр', 'музык', 'концерт', 'выставк', 'искусство', 'книг', 'литератур', 'опер', 'балет', 'стадион']):
        cats.append('culture')
    
    # Общество
    if any(w in full for w in ['школ', 'больниц', 'здрав', 'образован', 'строител', 'дорог', 'библиотек', 'района', 'сельск', 'животновод']):
        cats.append('society')
    
    # Политика
    if any(w in full for w in ['депутат', 'министр', 'губернатор', 'власт', 'правител', 'администра', 'суверен', 'декларац', 'закон']):
        cats.append('politics')
    
    return cats if cats else ['other']

# Загружаем и анализируем
with open('chuvash_news_final.json', encoding='utf-8') as f:
    articles = json.load(f)

categories = defaultdict(list)
total = len(articles)

for i, art in enumerate(articles):
    if i % 5000 == 0:
        print(f"Обработано {i}/{total}...", flush=True)
    
    url = art.get('url', '')
    if url in excluded:
        continue
    
    if not art.get('text') or len(art['text']) < 50:
        continue
    
    score = score_article(art)
    if score < 0:
        continue
    
    cats = categorize(art['text'], art['title'])
    
    for cat in cats:
        categories[cat].append({
            'url': url,
            'date': art.get('date', ''),
            'title': art.get('title', ''),
            'text': art.get('text', ''),
            'score': score
        })

print(f"\nИтого обработано: {total}")

# Выбираем лучшие
results = {}
for cat in ['sport', 'culture', 'society', 'politics']:
    if cat in categories:
        sorted_arts = sorted(categories[cat], key=lambda x: x['score'], reverse=True)
        results[cat] = sorted_arts[:5]
        print(f"\n{cat}: найдено {len(categories[cat])}, показываем {min(5, len(sorted_arts))}")

# Выводим результаты
for cat in ['sport', 'culture', 'society', 'politics']:
    if cat in results:
        print(f"\n{'='*80}")
        print(f"{cat.upper()}")
        print('='*80)
        for i, art in enumerate(results[cat], 1):
            print(f"\n{i}. {art['url']}")
            print(f"   Дата: {art['date']}")
            print(f"   Заголовок: {art['title']}")
            print(f"   Текст: {art['text'][:250]}...")
            print(f"   Оценка: {art['score']}")

