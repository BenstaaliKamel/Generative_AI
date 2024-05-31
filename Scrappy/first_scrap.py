import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader



def get_all_href_links(url):
    """
    Récupère tous les liens href d'une page web donnée et sauvegarde le contenu HTML localement.

    :param url: L'URL de la page web
    :return: Une liste contenant tous les liens href
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Save the HTML content to a local file
        with open("page.html", "w", encoding="utf-8") as file:
            file.write(response.text)

        # Read the saved HTML file
        with open("page.html", "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, 'html.parser')
            href_links = [link.get('href') for link in soup.find_all('a', href=True)]
            return href_links
    except requests.RequestException as e:
        print(f"Une erreur s'est produite lors de la requête HTTP : {e}")
        return []
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return []

# Test de la fonction avec une URL
# url = "https://webscraper.io/test-sites/e-commerce/allinone"  # Remplacez ceci par l'URL de votre choix
# href_links = get_all_href_links(url)
# print("Liens href trouvés sur la page :", href_links)

loader = WebBaseLoader("https://webscraper.io/test-sites/e-commerce/allinone")
docs = loader.load()
print(docs[0].page_content[:500])
