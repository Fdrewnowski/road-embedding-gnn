

# Model params
ENCODER = ['gcn', 'gat', 'gin']
ACTIVATION = ['prelu']#, 'relu'] relu used once
DECODER = ['gcn', 'gat', 'gin']
NUM_HIDDEN = [128, 256, 512, 1024]
NUM_OUT = [32, 64, 128, 256, 512]
NUM_LAYERS = [2,3,4]
IN_DROP = [0.1, 0.2]

# training params
MAX_EPOCHS = [500]

## optimizer params
OPTIMIZER = ['adam', 'sgd']
WEIGHT_DECAY = [0, 2e-4] # yes or very small
LR = [0.01, 0.001, 0.00015]

TRAINING_SET = [
                # DENMARK 
                "Copenhagen Municipality, Region Stołeczny, Dania",
                "Gmina Aarhus, Jutlandia Środkowa, Dania",
                "Odense Kommune, Dania Południowa, Dania",
                "Gmina Aalborg, Jutlandia Północna, Dania",
                "Frederiksberg Municipality, Region Stołeczny, Dania",
                "Gimina Gentofte, Region Stołeczny, Dania",
                "Gmina Esbjerg, Dania Południowa, Dania",
                "Gmina Gladsaxe, Region Stołeczny, Dania",
                "Gmina Holstebro, Holstebro Municipality, Dania",
                "Gmina Randers, Jutlandia Środkowa, Dania",
                "Gmina Kolding, Dania Południowa, Dania",
                # POLAND
                "Warszawa, województwo mazowieckie, Polska",
                "Łódź, województwo łódzkie, Polska",
                "Kraków, województwo małopolskie, Polska",
                "Poznań, powiat poznański, województwo wielkopolskie, Polska",
                "Szczecin, województwo zachodniopomorskie, Polska",
                "Bydgoszcz, województwo kujawsko-pomorskie, Polska",
                "Lublin, województwo lubelskie, Polska",
                "Białystok, powiat białostocki, województwo podlaskie, Polska",
                "Gdynia, województwo pomorskie, Polska",
                "Katowice, Górnośląsko-Zagłębiowska Metropolia, województwo śląskie, Polska",
                # BELGIUM 
                "Antwerpia, Flanders, Belgia",
                "Ghent, Gent, Flandria Wschodnia, Flanders, Belgia",
                "Charleroi, Hainaut, Walonia, Belgia",
                "Liège, Walonia, 4000, Belgia",
                "Bruksela, Brussels, Brussels-Capital, Belgia",
                "Schaerbeek - Schaarbeek, Brussels-Capital, Belgia",
                "Anderlecht, Brussels-Capital, 1070, Belgia",
                "Brugia, Flandria Zachodnia, Flanders, Belgia",
                "Namur, Walonia, Belgia",

                "Utrecht, Niderlandy, Holandia",
                "Birmingham, West Midlands Combined Authority, Anglia, Wielka Brytania"
                "Lyon, Métropole de Lyon, Rhône, Owernia-Rodan-Alpy, Francja metropolitalna, Francja",
                
                "Brno, okres Brno-město, Kraj południowomorawski, Południowo-wschodni, Czechy",
                "Ostrawa, Powiat Ostrawa-miasto, Kraj morawsko-śląski, Czechy",
                "Oslo, Norwegia",
                "Stockholms kommun, Stockholm County, Szwecja",
                "Malmo, Malmö kommun, Skåne County, Szwecja",
                "Helsinki, Helsinki sub-region, Uusimaa, Southern Finland, Mainland Finland, Finlandia",

                "Berno, Bern-Mittelland administrative district, Bernese Mittelland administrative region, Berno, Szwajcaria",
                "Zurych, District Zurich, Zurych, Szwajcaria",
                
                "Brema, Free Hanseatic City of Bremen, Niemcy",
                "Hanower, Region Hannover, Dolna Saksonia, Niemcy",
                "Essen, Nadrenia Północna-Westfalia, Niemcy",

                "Mediolan, Lombardia, Włochy",
                "Florencja, Metropolitan City of Florence, Toskania, Włochy",
                "Sewilla, Sevilla, Andaluzja, Hiszpania",
                "Saragossa, Zaragoza, Saragossa, Aragonia, Hiszpania",
                "Walencja, Comarca de València, Walencja, Wspólnota Walencka, Hiszpania",
                "Porto, Portugalia"
                ]

VALIDATION_SET = ["Wrocław, województwo dolnośląskie, Polska",
                  "Gdańsk, województwo pomorskie, Polska",
                  "Reims, Marna, Grand Est, Francja metropolitalna, 51100, Francja",
                  "Bolonia, Emilia-Romania, Włochy",
                  "Eindhoven, Brabancja Północna, Niderlandy, Holandia"
                  ]
