"""Alcohol domain knowledge: drink categories, brands, traps, and negations.

Every term is lowercase ASCII. Multi-word entries are matched as phrases.

Design notes
------------
* STRONG terms are unambiguous alcohol references (``bourbon``, ``heineken``).
* AMBIGUOUS terms usually mean alcohol but have a common non-alcohol sense
  (``port`` the harbour, ``rye`` the bread, ``corona`` the virus). A match on
  one of these is reported with lower confidence.
* TRAP words appear in alcohol ads *and* in the non-alcohol class of this
  dataset (``brew`` lemonade, espresso ``bar``, ``nightlife``). They are never
  used as evidence by the lexicon.
* NEGATION phrases turn a nearby alcohol word into a non-alcohol product
  (``ginger ale``, ``root beer``, ``alcohol-free``). They are removed from the
  text before matching.
"""

from __future__ import annotations

BEER = [
    "beer", "beers", "lager", "lagers", "pilsner", "pilsener", "stout", "ipa",
    "pale ale", "ale", "ales", "bock", "hefeweizen", "weissbier", "witbier",
    "porter", "draught beer", "draft beer", "craft beer", "brewski", "cider",
    "hard cider", "hard seltzer", "shandy", "mead", "malt liquor",
]

WINE = [
    "wine", "wines", "red wine", "white wine", "rose wine", "chardonnay",
    "merlot", "cabernet", "sauvignon", "pinot", "pinot noir", "pinot grigio",
    "riesling", "shiraz", "syrah", "malbec", "zinfandel", "prosecco",
    "champagne", "cava", "sangria", "sherry", "vermouth", "madeira", "marsala",
    "moscato", "bordeaux", "chianti", "rioja", "lambrusco", "tempranillo",
    "sommelier", "vino",
]

SPIRITS = [
    "whiskey", "whisky", "whiskies", "bourbon", "scotch", "vodka", "gin",
    "rum", "tequila", "mezcal", "brandy", "cognac", "armagnac", "absinthe",
    "schnapps", "liqueur", "liquor", "liquors", "sake", "soju", "baijiu",
    "ouzo", "grappa", "pisco", "cachaca", "moonshine", "arak", "raki",
    "sambuca", "limoncello", "amaretto", "calvados", "single malt",
]

COCKTAILS = [
    "cocktail", "cocktails", "martini", "margarita", "mojito", "daiquiri",
    "cosmopolitan", "negroni", "mimosa", "bellini", "spritz", "aperol spritz",
    "old fashioned", "manhattan cocktail", "pina colada", "bloody mary",
    "mai tai", "long island iced tea", "moscow mule", "whiskey sour",
    "gin and tonic", "gin tonic", "highball",
]

GENERIC = [
    "alcohol", "alcoholic", "booze", "boozy", "hooch", "brewery", "breweries",
    "distillery", "winery", "vineyard", "bartender", "happy hour", "hangover",
    "tipsy", "drunk", "beer garden", "wine bar", "pub crawl", "open bar",
    "on tap", "keg", "kegs", "pint of beer",
]

BRANDS = [
    # beer
    "budweiser", "bud light", "coors", "coors light", "miller lite",
    "miller high life", "heineken", "stella artois", "guinness", "carlsberg",
    "becks", "amstel", "michelob", "modelo", "pabst", "blue moon",
    "sam adams", "samuel adams", "peroni", "asahi", "sapporo", "kirin",
    "tsingtao", "dos equis", "pacifico", "hoegaarden", "leffe", "newcastle",
    "fosters", "tecate", "goldstar", "maccabee", "grolsch", "kronenbourg",
    "san miguel", "red stripe", "singha", "tiger beer", "efes", "duvel",
    "chimay", "weihenstephaner", "paulaner", "erdinger", "busch", "natty light",
    "white claw", "truly seltzer",
    # vodka
    "smirnoff", "absolut", "grey goose", "belvedere", "ketel one", "skyy",
    "stolichnaya", "stoli", "titos", "ciroc", "finlandia", "svedka",
    # whiskey
    "jack daniels", "jim beam", "jameson", "johnnie walker", "chivas",
    "chivas regal", "glenfiddich", "glenlivet", "macallan", "makers mark",
    "wild turkey", "crown royal", "fireball", "bulleit", "woodford reserve",
    "jack daniel", "dewars", "ballantines", "famous grouse", "laphroaig",
    "lagavulin", "talisker", "buffalo trace", "evan williams",
    # rum / gin / tequila / cognac
    "bacardi", "captain morgan", "havana club", "kraken", "sailor jerry",
    "malibu rum", "bombay sapphire", "tanqueray", "beefeater", "gordons gin",
    "hendricks", "jose cuervo", "patron", "don julio", "casamigos",
    "hennessy", "remy martin", "courvoisier", "martell",
    # liqueurs / wine / champagne
    "jagermeister", "baileys", "kahlua", "cointreau", "grand marnier",
    "disaronno", "midori", "southern comfort", "campari", "aperol",
    "moet", "moet chandon", "dom perignon", "veuve clicquot", "yellow tail",
    "barefoot wine", "yarden", "barkan", "carmel winery", "martini rossi",
]

# Alternate names and spellings of brands already listed above. Matched as
# exact terms. Short nicknames that are also ordinary words or given names
# ("bud", "jd", "henny", "stella") are left out on purpose.
BRAND_SYNONYMS = [
    "gray goose",       # grey goose
    "johnny walker",    # johnnie walker
    "maccabi",          # maccabee
    "gold star",        # goldstar
    "jager",            # jagermeister
    "jaeger",
    "jaegermeister",
]

# Words whose alcohol sense is common but not the only sense.
AMBIGUOUS = {
    "port", "rye", "corona", "porter", "cava", "patron", "busch", "manhattan",
    "kraken", "mead", "stoli", "cider", "sake", "martini", "fosters",
    "newcastle", "madeira", "bordeaux", "vino", "moet", "absolut", "keg",
    "kegs", "tipsy", "drunk", "hangover", "highball",
}

# Appear on both sides of this dataset; never used as evidence by the lexicon.
TRAPS = {
    "brew", "brews", "brewed", "bar", "bars", "pub", "nightlife", "party",
    "celebration", "cheers", "toast", "sparkling", "reserve", "vintage", "aged",
    "spirit", "spirits", "shot", "shots", "cellar", "tap", "premium", "imported",
    "smooth", "flavor", "flavour", "taste", "event", "menu", "special",
    "limited", "edition", "drink", "drinks", "sip", "pour", "round", "club",
    "lounge", "tavern", "saloon", "night", "weekend", "happy", "hour",
}

# Modifiers that cancel the drink word that follows them ("alcohol-free beer",
# "non-alcoholic wine"). The modifier and the next NEGATION_SCOPE tokens are
# removed before matching.
NEGATION_MODIFIERS = [
    "non alcoholic", "alcohol free", "zero alcohol", "zero proof", "no alcohol",
    "0.0", "0% alcohol", "dealcoholized", "alcohol removed", "virgin",
]
NEGATION_SCOPE = 2

# Complete phrases that contain an alcohol word but are not alcohol.
NEGATIONS = [
    "mocktail", "mocktails", "root beer", "ginger beer", "ginger ale", "birch beer", "butterbeer",
    "wine gum", "wine gums", "rum raisin", "beer battered", "grape juice",
    "sparkling cider", "apple cider vinegar", "cider vinegar", "wine vinegar",
    "red wine vinegar", "rice wine vinegar", "sake salmon",
]

# Non-alcohol drinks and other "vice" products seen in the data. Not used for
# the decision; used to explain predictions and to protect words from fuzzy
# matching against alcohol terms.
NON_ALCOHOL_DRINKS = {
    "water", "milk", "coffee", "espresso", "latte", "cappuccino", "tea",
    "iced tea", "herbal tea", "green tea", "lemonade", "juice", "soda", "cola",
    "sprite", "energy drink", "smoothie", "kombucha", "cocoa", "hot chocolate",
    "cannabis", "cbd", "weed", "hemp", "vape", "tobacco", "cigar", "cigarette",
    "matcha", "chai", "sparkling water", "mineral water", "coconut water",
    "milkshake", "frappe", "seltzer", "tonic", "ginger", "lemon", "lime",
}

# Common words that are one edit away from an alcohol term and must never be
# fuzzy-matched to it (e.g. brand -> brandy, later -> lager).
FUZZY_BLOCKLIST = {
    "brand", "brands", "candy", "handy", "sandy", "randy", "later", "layer",
    "laser", "eager", "rider", "wider", "cinder", "ginger", "bourse", "wines",
    "whisk", "vodafone", "spider", "sider", "stouts", "court", "sport", "sports",
    "short", "porch", "coffee", "toffee", "water", "waiter", "winner",
    "dinner", "diner", "liquid", "ciders", "scotland", "scottish", "margin",
    "martin", "barber", "bourbonne", "champion", "champions", "absolute",
    "absolutely", "patrol", "patron", "moist", "chivalry", "corolla", "coronas",
    "heinz", "smirk", "baccarat", "bacon", "tequilla", "cabin", "cabinet",
    "reserve", "reserved", "flavor", "flavors", "premium", "special", "limited",
}

# The ad-template words the caption generator glues onto drink words
# ("importedrum", "champagnemenu", "offerbacardi"). They are peeled off the
# start / end of long tokens so the drink word underneath can be matched.
TEMPLATE_WORDS = sorted({
    "try", "dont", "miss", "check", "out", "get", "ready", "for", "grab", "now",
    "enjoy", "looking", "experience", "explore", "offer", "discount", "limited",
    "edition", "premium", "imported", "special", "event", "menu", "taste",
    "flavor", "flavour", "smooth", "party", "nightlife", "celebration", "reserve",
    "brew", "bar", "the",
}, key=len, reverse=True)

# Ordinary words that contain an alcohol term; a glued-word (substring) match
# inside one of these tokens is ignored (absolutely -> absolut).
SUBSTRING_BLOCKLIST = {
    "absolute", "absolutely", "reporter", "reporters", "butterscotch",
    "hopscotch", "villager", "villagers", "decider", "twines", "sherryl",
    "important", "importer", "passport", "airport", "transport", "support",
    "supporter", "sporter", "portable", "stoutly", "ciderella", "brandywine",
    "cavalier", "cavalry", "ginseng", "virginia", "original", "begin",
    "margin", "engine", "rumble", "drum", "spectrum", "forum", "crumb",
    "beginner", "stale", "scale", "female", "tale", "sale", "pale", "male",
    "ginger", "lagerfeld", "realer", "dealer", "healer", "sealer", "vale",
}

STRONG_TERMS = sorted(
    {t for t in BEER + WINE + SPIRITS + COCKTAILS + GENERIC + BRANDS + BRAND_SYNONYMS}
    - AMBIGUOUS
)
ALL_TERMS = sorted(set(STRONG_TERMS) | AMBIGUOUS)


def term_category(term: str) -> str:
    """Return the lexicon category of ``term`` (used in explanations)."""
    for name, group in (
        ("brand", list(BRANDS) + list(BRAND_SYNONYMS)),
        ("beer", BEER),
        ("wine", WINE),
        ("spirit", SPIRITS),
        ("cocktail", COCKTAILS),
        ("generic", GENERIC),
    ):
        if term in group:
            return name
    return "ambiguous" if term in AMBIGUOUS else "unknown"
