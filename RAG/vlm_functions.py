import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import os
import pickle
from pathlib import Path

DOTA_HEROES = [
    "Abaddon", "Alchemist", "Ancient Apparition", "Anti-Mage", "Arc Warden", "Axe",
    "Bane", "Batrider", "Beastmaster", "Bloodseeker", "Bounty Hunter", "Brewmaster",
    "Bristleback", "Broodmother", "Centaur Warrunner", "Chaos Knight", "Chen", "Clinkz",
    "Clockwerk", "Crystal Maiden", "Dark Seer", "Dark Willow", "Dawnbreaker", "Dazzle",
    "Death Prophet", "Disruptor", "Doom", "Dragon Knight", "Drow Ranger", "Earth Spirit",
    "Earthshaker", "Elder Titan", "Ember Spirit", "Enchantress", "Enigma", "Faceless Void",
    "Grimstroke", "Gyrocopter", "Hoodwink", "Huskar", "Invoker", "Io", "Jakiro",
    "Juggernaut", "Keeper of the Light", "Kunkka", "Legion Commander", "Leshrac", "Lich",
    "Lifestealer", "Lina", "Lion", "Lone Druid", "Luna", "Lycan", "Magnus", "Marci",
    "Mars", "Medusa", "Meepo", "Mirana", "Monkey King", "Morphling", "Muerta",
    "Naga Siren", "Nature's Prophet", "Necrophos", "Night Stalker", "Nyx Assassin",
    "Ogre Magi", "Omniknight", "Oracle", "Outworld Destroyer", "Pangolier",
    "Phantom Assassin", "Phantom Lancer", "Phoenix", "Primal Beast", "Puck", "Pudge",
    "Pugna", "Queen of Pain", "Razor", "Riki", "Rubick", "Sand King", "Shadow Demon",
    "Shadow Fiend", "Shadow Shaman", "Silencer", "Skywrath Mage", "Slardar", "Slark",
    "Snapfire", "Sniper", "Spectre", "Spirit Breaker", "Storm Spirit", "Sven", "Techies",
    "Templar Assassin", "Terrorblade", "Tidehunter", "Timbersaw", "Tinker", "Tiny",
    "Treant Protector", "Troll Warlord", "Tusk", "Underlord", "Undying", "Ursa",
    "Vengeful Spirit", "Venomancer", "Viper", "Visage", "Void Spirit", "Warlock",
    "Weaver", "Windranger", "Winter Wyvern", "Witch Doctor", "Wraith King", "Zeus"
]

print(len(DOTA_HEROES))


def load_image(image_path):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    return image

def download_hero_icons():
    icons_dir = Path("dota_hero_icons")
    icons_dir.mkdir(exist_ok=True)
    
    hero_url_mapping = {
        "Abaddon": "abaddon",
        "Alchemist": "alchemist",
        "Ancient Apparition": "ancient_apparition",
        "Anti-Mage": "antimage",
        "Arc Warden": "arc_warden",
        "Axe": "axe",
        "Bane": "bane",
        "Batrider": "batrider",
        "Beastmaster": "beastmaster",
        "Bloodseeker": "bloodseeker",
        "Bounty Hunter": "bounty_hunter",
        "Brewmaster": "brewmaster",
        "Bristleback": "bristleback",
        "Broodmother": "broodmother",
        "Centaur Warrunner": "centaur",
        "Chaos Knight": "chaos_knight",
        "Chen": "chen",
        "Clinkz": "clinkz",
        "Clockwerk": "rattletrap",
        "Crystal Maiden": "crystal_maiden",
        "Dark Seer": "dark_seer",
        "Dark Willow": "dark_willow",
        "Dawnbreaker": "dawnbreaker",
        "Dazzle": "dazzle",
        "Death Prophet": "death_prophet",
        "Disruptor": "disruptor",
        "Doom": "doom_bringer",
        "Dragon Knight": "dragon_knight",
        "Drow Ranger": "drow_ranger",
        "Earth Spirit": "earth_spirit",
        "Earthshaker": "earthshaker",
        "Elder Titan": "elder_titan",
        "Ember Spirit": "ember_spirit",
        "Enchantress": "enchantress",
        "Enigma": "enigma",
        "Faceless Void": "faceless_void",
        "Grimstroke": "grimstroke",
        "Gyrocopter": "gyrocopter",
        "Hoodwink": "hoodwink",
        "Huskar": "huskar",
        "Invoker": "invoker",
        "Io": "wisp",
        "Jakiro": "jakiro",
        "Juggernaut": "juggernaut",
        "Keeper of the Light": "keeper_of_the_light",
        "Kunkka": "kunkka",
        "Legion Commander": "legion_commander",
        "Leshrac": "leshrac",
        "Lich": "lich",
        "Lifestealer": "life_stealer",
        "Lina": "lina",
        "Lion": "lion",
        "Lone Druid": "lone_druid",
        "Luna": "luna",
        "Lycan": "lycan",
        "Magnus": "magnataur",
        "Marci": "marci",
        "Mars": "mars",
        "Medusa": "medusa",
        "Meepo": "meepo",
        "Mirana": "mirana",
        "Monkey King": "monkey_king",
        "Morphling": "morphling",
        "Muerta": "muerta",
        "Naga Siren": "naga_siren",
        "Nature's Prophet": "furion",
        "Necrophos": "necrolyte",
        "Night Stalker": "night_stalker",
        "Nyx Assassin": "nyx_assassin",
        "Ogre Magi": "ogre_magi",
        "Omniknight": "omniknight",
        "Oracle": "oracle",
        "Outworld Destroyer": "obsidian_destroyer",
        "Pangolier": "pangolier",
        "Phantom Assassin": "phantom_assassin",
        "Phantom Lancer": "phantom_lancer",
        "Phoenix": "phoenix",
        "Primal Beast": "primal_beast",
        "Puck": "puck",
        "Pudge": "pudge",
        "Pugna": "pugna",
        "Queen of Pain": "queenofpain",
        "Razor": "razor",
        "Riki": "riki",
        "Rubick": "rubick",
        "Sand King": "sand_king",
        "Shadow Demon": "shadow_demon",
        "Shadow Fiend": "nevermore",
        "Shadow Shaman": "shadow_shaman",
        "Silencer": "silencer",
        "Skywrath Mage": "skywrath_mage",
        "Slardar": "slardar",
        "Slark": "slark",
        "Snapfire": "snapfire",
        "Sniper": "sniper",
        "Spectre": "spectre",
        "Spirit Breaker": "spirit_breaker",
        "Storm Spirit": "storm_spirit",
        "Sven": "sven",
        "Techies": "techies",
        "Templar Assassin": "templar_assassin",
        "Terrorblade": "terrorblade",
        "Tidehunter": "tidehunter",
        "Timbersaw": "shredder",
        "Tinker": "tinker",
        "Tiny": "tiny",
        "Treant Protector": "treant",
        "Troll Warlord": "troll_warlord",
        "Tusk": "tusk",
        "Underlord": "abyssal_underlord",
        "Undying": "undying",
        "Ursa": "ursa",
        "Vengeful Spirit": "vengefulspirit",
        "Venomancer": "venomancer",
        "Viper": "viper",
        "Visage": "visage",
        "Void Spirit": "void_spirit",
        "Warlock": "warlock",
        "Weaver": "weaver",
        "Windranger": "windrunner",
        "Winter Wyvern": "winter_wyvern",
        "Witch Doctor": "witch_doctor",
        "Wraith King": "skeleton_king",
        "Zeus": "zuus"
    }
    
    downloaded = 0
    for hero in DOTA_HEROES:
        if hero in hero_url_mapping:
            hero_url = hero_url_mapping[hero]
        else:
            hero_url = hero.replace(" ", "_").replace("'", "").replace("-", "_").lower()
        
        url = f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{hero_url}.png"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(icons_dir / f"{hero}.png", "wb") as f:
                    f.write(response.content)
                print(f"✅ Скачан: {hero} -> {hero_url}")
                downloaded += 1
            else:
                print(f"❌ Ошибка {hero} ({hero_url}): {response.status_code}")
        except Exception as e:
            print(f"❌ Ошибка {hero}: {e}")
    
    print(f"\nСкачано {downloaded} иконок из {len(DOTA_HEROES)}")
    return icons_dir

def create_hero_database(force_recreate=False):
    if not force_recreate and os.path.exists("hero_embeddings.pkl"):
        print("База данных уже существует. Используйте force_recreate=True для пересоздания.")
        with open("hero_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    
    icons_dir = Path("dota_hero_icons")
    if not icons_dir.exists() or len(list(icons_dir.glob("*.png"))) < 50:
        print("Скачиваем иконки героев...")
        icons_dir = download_hero_icons()
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    hero_embeddings = {}
    
    print("Создаем эмбеддинги для иконок...")
    for hero in DOTA_HEROES:
        icon_path = icons_dir / f"{hero}.png"
        if icon_path.exists():
            try:
                image = Image.open(icon_path)
                inputs = processor(images=image, return_tensors="pt")
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    hero_embeddings[hero] = image_features.cpu().numpy()
                
                print(f"Обработан: {hero}")
            except Exception as e:
                print(f"Ошибка {hero}: {e}")
    
    with open("hero_embeddings.pkl", "wb") as f:
        pickle.dump(hero_embeddings, f)
    
    print(f"\nБаза данных создана: {len(hero_embeddings)} героев")
    return hero_embeddings

def recognize_heroes(image_path):
    if not os.path.exists("hero_embeddings.pkl"):
        print("База данных не найдена. Создаем...")
        hero_embeddings = create_hero_database()
    else:
        with open("hero_embeddings.pkl", "rb") as f:
            hero_embeddings = pickle.load(f)
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()
    
    similarities = {}
    for hero, hero_embedding in hero_embeddings.items():
        similarity = np.dot(query_embedding, hero_embedding.T).item()
        similarities[hero] = similarity
    
    sorted_heroes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    top_heroes = [hero for hero, score in sorted_heroes[:3]]
    top_scores = [score for hero, score in sorted_heroes[:3]]
    
    return {
        "heroes": top_heroes,
        "scores": top_scores,
        "best_match": top_heroes[0] if top_scores[0] > 0.3 else "Unknown"
    }

def test_recognition(image_path):
    print(f"Тестируем изображение: {image_path}")
    try:
        result = recognize_heroes(image_path)
        print(f"Лучший матч: {result['best_match']}")
        print(f"Топ-3 героя: {result['heroes']}")
        print(f"Уверенность: {[f'{score:.3f}' for score in result['scores']]}")
        return result
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def force_recreate_database():
    
    if os.path.exists("hero_embeddings.pkl"):
        os.remove("hero_embeddings.pkl")
        print("Удален hero_embeddings.pkl")
    
    if os.path.exists("dota_hero_icons"):
        import shutil
        shutil.rmtree("dota_hero_icons")
        print("Удалена папка dota_hero_icons")
    
    print("🔄 Пересоздаем базу данных...")
    return create_hero_database(force_recreate=True)

def analyze_image_with_rag(image_path, rag_store):
    """
    Полный пайплайн: изображение -> распознавание героев -> RAG -> ответ
    """
    print(f"Анализируем изображение: {image_path}")
    
    recognition_result = recognize_heroes(image_path)
    detected_heroes = recognition_result['heroes']
    
    print(f"Обнаружены герои: {detected_heroes}")
    
    if recognition_result['best_match'] == "Unknown":
        return "Не удалось распознать героев на изображении"
    
    all_info = []
    for hero in detected_heroes[:2]: 
        print(f"Ищем информацию о {hero}...")
        rag_results = rag_store.search(f"Who is {hero}? What are their abilities and role?", n_results=3)
        
        if rag_results['documents']:
            hero_info = "\n".join(rag_results['documents'][0])
            all_info.append(f"**{hero}:**\n{hero_info}")
    
    if all_info:
        response = f"На изображении обнаружены герои: {', '.join(detected_heroes)}\n\n"
        response += "\n\n".join(all_info)
        return response
    else:
        return f"Обнаружены герои: {', '.join(detected_heroes)}, но не удалось найти подробную информацию"

def test_download_specific_hero(hero_name):
    hero_url_mapping = {
        "Abaddon": "abaddon", "Alchemist": "alchemist", "Ancient Apparition": "ancient_apparition",
        "Anti-Mage": "antimage", "Arc Warden": "arc_warden", "Axe": "axe", "Bane": "bane",
        "Batrider": "batrider", "Beastmaster": "beastmaster", "Bloodseeker": "bloodseeker",
        "Bounty Hunter": "bounty_hunter", "Brewmaster": "brewmaster", "Bristleback": "bristleback",
        "Broodmother": "broodmother", "Centaur Warrunner": "centaur", "Chaos Knight": "chaos_knight",
        "Chen": "chen", "Clinkz": "clinkz", "Clockwerk": "rattletrap", "Crystal Maiden": "crystal_maiden",
        "Dark Seer": "dark_seer", "Dark Willow": "dark_willow", "Dawnbreaker": "dawnbreaker",
        "Dazzle": "dazzle", "Death Prophet": "death_prophet", "Disruptor": "disruptor",
        "Doom": "doom_bringer", "Dragon Knight": "dragon_knight", "Drow Ranger": "drow_ranger",
        "Earth Spirit": "earth_spirit", "Earthshaker": "earthshaker", "Elder Titan": "elder_titan",
        "Ember Spirit": "ember_spirit", "Enchantress": "enchantress", "Enigma": "enigma",
        "Faceless Void": "faceless_void", "Grimstroke": "grimstroke", "Gyrocopter": "gyrocopter",
        "Hoodwink": "hoodwink", "Huskar": "huskar", "Invoker": "invoker", "Io": "wisp",
        "Jakiro": "jakiro", "Juggernaut": "juggernaut", "Keeper of the Light": "keeper_of_the_light",
        "Kunkka": "kunkka", "Legion Commander": "legion_commander", "Leshrac": "leshrac",
        "Lich": "lich", "Lifestealer": "life_stealer", "Lina": "lina", "Lion": "lion",
        "Lone Druid": "lone_druid", "Luna": "luna", "Lycan": "lycan", "Magnus": "magnataur",
        "Marci": "marci", "Mars": "mars", "Medusa": "medusa", "Meepo": "meepo",
        "Mirana": "mirana", "Monkey King": "monkey_king", "Morphling": "morphling",
        "Muerta": "muerta", "Naga Siren": "naga_siren", "Nature's Prophet": "furion",
        "Necrophos": "necrolyte", "Night Stalker": "night_stalker", "Nyx Assassin": "nyx_assassin",
        "Ogre Magi": "ogre_magi", "Omniknight": "omniknight", "Oracle": "oracle",
        "Outworld Destroyer": "obsidian_destroyer", "Pangolier": "pangolier",
        "Phantom Assassin": "phantom_assassin", "Phantom Lancer": "phantom_lancer",
        "Phoenix": "phoenix", "Primal Beast": "primal_beast", "Puck": "puck", "Pudge": "pudge",
        "Pugna": "pugna", "Queen of Pain": "queenofpain", "Razor": "razor", "Riki": "riki",
        "Rubick": "rubick", "Sand King": "sand_king", "Shadow Demon": "shadow_demon",
        "Shadow Fiend": "nevermore", "Shadow Shaman": "shadow_shaman", "Silencer": "silencer",
        "Skywrath Mage": "skywrath_mage", "Slardar": "slardar", "Slark": "slark",
        "Snapfire": "snapfire", "Sniper": "sniper", "Spectre": "spectre",
        "Spirit Breaker": "spirit_breaker", "Storm Spirit": "storm_spirit", "Sven": "sven",
        "Techies": "techies", "Templar Assassin": "templar_assassin", "Terrorblade": "terrorblade",
        "Tidehunter": "tidehunter", "Timbersaw": "shredder", "Tinker": "tinker", "Tiny": "tiny",
        "Treant Protector": "treant", "Troll Warlord": "troll_warlord", "Tusk": "tusk",
        "Underlord": "abyssal_underlord", "Undying": "undying", "Ursa": "ursa",
        "Vengeful Spirit": "vengefulspirit", "Venomancer": "venomancer", "Viper": "viper",
        "Visage": "visage", "Void Spirit": "void_spirit", "Warlock": "warlock", "Weaver": "weaver",
        "Windranger": "windrunner", "Winter Wyvern": "winter_wyvern", "Witch Doctor": "witch_doctor",
        "Wraith King": "skeleton_king", "Zeus": "zuus"
    }
    
    if hero_name in hero_url_mapping:
        hero_url = hero_url_mapping[hero_name]
    else:
        hero_url = hero_name.replace(" ", "_").replace("'", "").replace("-", "_").lower()
    
    url = f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{hero_url}.png"
    print(f"Тестируем URL: {url}")
    
    try:
        response = requests.get(url)
        print(f"Статус: {response.status_code}")
        if response.status_code == 200:
            print("URL работает!")
            return url
        else:
            print("URL не работает")
            return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

def restart_training():
    force_recreate_database()
    print("Обучение завершено")

def get_hero_name_from_image(image_path):

    result = recognize_heroes(image_path)
    return result['best_match']

