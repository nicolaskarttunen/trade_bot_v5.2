# Strategiatutkimus erillään virhekorjauksista

**Kirjattu 23.10.2026. Mitään alla olevaa hyötyä ei ole osoitettu.**
Kauppaparametreja ei optimoitu tässä korjaustyössä. `config.json` säilyi samana.
Piirteiden ja laskennan korjaukset muuttavat tuloksia, joten vanhaa tuottoa ei
voi verrata uuteen ja nimetä erotusta suodattimen ansioksi.

## SPY +0,10 %: mitä voidaan nyt päätellä?

Suodattimen hyöty on avoin. Rajaa on aiemmin tarkasteltu tulosten perusteella,
ja 1.9.–19.10.2026 on jo nähty kehitysjakso. Käyttäjän vastauksen perusteella
myös 20.–23.10.2026 käsitellään nähtynä tai epävarmana, ei uutena holdoutina.
Nykyinen ajuri on tarkoituksella kehitysvertailu; se ei väitä tuottavansa
riippumatonta näyttöä. Täysi data ei ollut tämän työn aikana saatavilla.

Vertailun täsmällinen määritelmä:

- Sama koulutus, malli, symbolijärjestys, osakejoukko, kulut, positiorajat ja
  markkinadata molemmille haaroille. Vain portti päällä/pois vaihtuu.
- Kiinteä portti: SPY:n **09.30–09.35 ET** kynttilän `close/open - 1 >= 0.001`.
  Kyseessä ei ole muutos edellisen päivän päätöksestä. Saatavilla aikaisintaan
  09.35; koskaan sitä ei käytetä aikaisempaan opening-/OPG-päätökseen.
- Ensimmäinen kelvollinen sisäänmeno on 09.35, viimeinen 15.25 tai aiemmin
  lyhennetyn session mukaan. Uutiset neutraaleina molemmissa haaroissa.
- Kaikki sessiot, myös nollakauppapäivät, säilytetään. Osakkeita tai päiviä ei
  poisteta häviöiden tai heikon suodatintuloksen perusteella.
- Ensisijainen mitta: paritettu päivittäisten nettotuottojen erotus
  `(gate_pnl/gate_start_equity) - (base_pnl/base_start_equity)`.
  Lisäksi kokonaisnettotuotto, MTM-drawdown, kauppamäärä, portin sallimat päivät,
  altistus, kustannukset ja tuloksen keskittyminen osakkeisiin/päiviin.
- `research_validation.py` antaa kuvailevan 95 % moving-block-bootstrap-välin:
  5 session lohkot, 2 000 toistoa, siemen 20261023, vähintään 10 päivää.
  Tämä ei korjaa aiempaa mallinvalintaa eikä tee nähdystä datasta OOS:ää.
- Puuttuva SPY-avausbaari, puuttuva sessio tai avoimen position hintapolun aukko
  keskeyttää hyväksyttävän ajon. Huonoja havaintoja ei vain pudoteta tuloksesta.

### Tulevan riippumattoman arvioinnin ehdotus

Aikaisin mahdollinen uusi alku on 26.10.2026, jos jaksoa ei ehditä tarkastella
ennen koodin, data-/mallisopimuksen ja kokeen jäädyttämistä. Ehdotettu kiinteä
loppu on 30.4.2027. Tätä suunnitelmaa ei ole ajastettu eikä tulevaa testiä ajettu.

Ennen jakson alkua tallenna erilliseen rekisteriin tarkka commit, config,
malli/tarkistussumma, koulutusraja, point-in-time-osakejoukko, feed, kalenteri,
kustannusoletukset ja kaikki aiemmat kokeet. Älä muuta kynnystä, lopeta koetta
voiton kohdalla tai tarkastele porttiversioiden tuottoeroa jatkuvasti.
Toiminnan ja datan laatua saa valvoa tuloserosta riippumatta.

Raportoi ero epävarmuuksineen sovittuna lopetuspäivänä. Positiivinen
piste-estimaatti ei yksin riitä. Jos epävarmuus kattaa nollan, kauppoja on vähän
tai tulos keskittyy yhteen tapahtumaan, johtopäätös on **epäselvä**, ei
suodattimen vahvistaminen. Julkaise myös negatiivinen tulos. Jos ajanjakso tai
koodi muuttuu virhekorjauksen takia, kirjaa poikkeama ja älä nimeä jälkikäteen
valittua osajaksoa alkuperäiseksi kokeeksi.

Kulu- ja viiveherkkyydet ovat etukäteen nimetyt erilliset stressitestit, eivät
keinot valita tuottoisin kustannusarvo. Nykyinen ajuri ei vielä simuloi
intrabaarin todellista viivettä tai osatäyttöjä.

## Uutisanalyysin luotettava testi

Yhteisen numeropiirteen oikeellisuus on eri asia kuin uutistulkinnan laatu.
Testit vahvistavat ajoituksen, vaimennuksen, neutraalin tilan, duplikaatin
vaikutuksettomuuden ja saman tuloksen erä-/yksittäisajossa. Ne eivät todista
avainsanalaskurin ymmärtävän otsikoita, negaatioita, ohjeistusta tai yrityksen
kannalta olennaista uutta tietoa.

1. **Muuttumaton tapahtumaloki.** Säilytä id, versio, symbolit, alkuperäinen
   julkaisu- ja päivitysaika, ensimmäinen todellinen vastaanottoaika, lähde,
   otsikko/teksti, pisteytysmalli/versio sekä lasketut pisteet. Päivitys on uusi
   tapahtuma; vanhaa tekstiä ei korvata taaksepäin. Tallenna myös katkokset ja
   kattavuus. Älä täydennä vanhoja vastaanottoaikoja nykyhetkestä arvaten.
2. **Manuaalisesti tarkistettu otos.** Valitse ennen tulosten katsomista
   tapahtuma- ja päiväryhmittäinen otos, mukana neutraalit, negaatiot,
   ristiriitaiset otsikot, ticker-virheet, uudelleenjulkaisut ja korjaukset.
   Arvioi relevanttius ja sentimentti sokkona myöhemmälle kurssikehitykselle.
   Erottele arvioijien erimielisyys mallivirheestä. Tallenna precision/recall,
   sekaannusmatriisi ja kattavuus, älä vain yhtä tarkkuuslukua.
3. **Aikajärjestyksessä tapahtuva replay.** Tapahtuma tulee näkyviin vasta
   vastaanotto-/päivitysajan jälkeen. Käytä samaa `news_scores`-funktiota kuin
   ennusteessa; skannaa baarin päättyessä. Testaa viiveet, puuttuva uutinen,
   puuttuva feed, väärä symboli ja sama tarina useana versiona erikseen.
4. **Erillinen lisäarvokoe.** Sama tarkistettu aineisto ja kaikki muu strategia,
   uutiset pois vastaan yhtenäinen uutispiirre. Kouluta molemmat versiot vain
   omalla koulutusjaksollaan; uutisia käyttävää mallia ei vain pakoteta nollille
   ja tulosta nimetä reiluksi ablaatioksi. Jaa tarinan versiot samaan
   ajalliseen ryhmään ja purgaa kurssiluokan horisontti.
5. **Koskematon arvio.** Lukitse tekstimalli, aikarajat, vaimennus, valinnat,
   kulut ja mittarit ennen loppujaksoa. Jos käytetään myöhemmin koulutettua
   kielimallia vanhojen uutisten pisteytykseen, sen koulutusmuistin mahdollinen
   tulevaisuustieto on lisärajoitus; mieluummin tapahtumahetkellä tallennetut
   pisteet ja eteenpäin kerätty koe.

Repon seitsemän uutis-CSV:n julkaisuajat eivät osoita, milloin paperibotti sai
tiedon. Niillä voidaan tehdä eksplisiittinen `publication_proxy`-tutkimus,
mutta ei vahvistaa vastaanottoviiveen jälkeistä uutisetua. `received`-tilan
käyttöönotto edellyttää kattavaa vastaanottolokia; legacy-tapahtumat ilman
vastaanottoaikaa hylätään. Uusi kirjattu vastaanottoaika kehityskoodissa ei
muuta Oraclella jo kerättyä historiaa.
