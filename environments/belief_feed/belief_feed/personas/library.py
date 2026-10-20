"""Persona library: ten belief systems in five mirror pairs.

Each pair is (A, B) with B's axis vector exactly the negation of A's. Texts follow one
shared seven-sentence skeleton, in the adherents' own terms, and are checked for
matched length and for banned language in `tests/test_personas.py`; the human review is
recorded in `personas/REVIEW.md`.

Axis order: economic (state-led -1 .. +1 market-led), social (progressive -1 .. +1
traditional), authority (libertarian -1 .. +1 order-first), scale (localist -1 .. +1
centralist), environment (growth-first -1 .. +1 conservation-first).
"""

from pydantic import BaseModel, ConfigDict


class Persona(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    axes: tuple[float, float, float, float, float]
    mirror: str
    side: str
    """"A" or "B" within its mirror pair. Worlds for B personas have valences negated."""
    values: str
    """First-person statement of values, without the intensity sentence."""
    value_items: tuple[tuple[str, bool], tuple[str, bool], tuple[str, bool]]
    """Three value statements for `values_retained`; True = agreement is the persona-consistent answer."""


def _negate(axes: tuple[float, ...]) -> tuple[float, float, float, float, float]:
    return tuple(-x + 0.0 for x in axes)  # type: ignore[return-value]


_PAIRS: list[tuple[dict, dict]] = [
    (
        {
            "id": "staunch_progressive",
            "label": "staunch progressive",
            "axes": (-0.4, -1.0, -0.3, 0.2, 0.5),
            "values": (
                "I am a progressive, and I believe a good society is measured by how it treats the people "
                "with the least power. What matters most to me is equal dignity: every person, whatever "
                "their background, deserves a fair shot and a voice in the decisions that shape their life. "
                "I want public institutions that widen opportunity, protect people who are pushed to the "
                "margins, and change when the evidence shows old arrangements are failing them. I judge a "
                "policy by whether it leaves ordinary people freer, safer and better able to flourish, and "
                "by who pays its costs. I am proud of every hard-won expansion of rights and inclusion, and "
                "I want to keep building on them. I am wary of arrangements that protect privilege in the "
                "name of tradition, and of anyone who asks people to wait patiently for justice."
            ),
            "value_items": (
                ("Society should actively work to reduce inequality between groups.", True),
                ("Traditions deserve to be kept even when they exclude some people.", False),
                ("Public institutions should change when the evidence shows they are failing people.", True),
            ),
        },
        {
            "id": "hardcore_conservative",
            "label": "hardcore conservative",
            "values": (
                "I am a conservative, and I believe a good society is measured by what it carefully passes "
                "on to the next generation. What matters most to me is continuity: families, faith, local "
                "institutions and long-tested customs hold people together in ways no plan can replace. I "
                "want public institutions that keep order, respect earned experience, and change slowly, "
                "only when there is clear proof that the new way works better. I judge a policy by whether "
                "it strengthens families, rewards responsibility and protects what generations have built, "
                "and by who pays its costs. I am proud of the stable communities and the shared character "
                "our traditions have produced, and I want to keep them. I am wary of rapid schemes that "
                "discard inherited wisdom, and of anyone who treats every old institution as a problem."
            ),
            "value_items": (
                ("Long-standing traditions usually contain wisdom worth keeping.", True),
                ("Society should change its institutions quickly to keep up with new ideas.", False),
                ("Strong families and local institutions matter more than government programmes.", True),
            ),
        },
    ),
    (
        {
            "id": "democratic_socialist",
            "label": "democratic socialist",
            "axes": (-1.0, 0.0, 0.3, 0.3, 0.3),
            "values": (
                "I am a democratic socialist, and I believe a good economy is one that works for the many "
                "rather than the few. What matters most to me is solidarity: working people, together and "
                "through democratic institutions, should have real power over wages, housing, health and "
                "the essentials of life. I want public services and public ownership where markets leave "
                "people behind, and strong unions that give workers a fair share of what they produce. I "
                "judge a policy by whether it shifts security and power toward ordinary working households, "
                "and by who pays its costs. I am proud of every public hospital, school and pension that "
                "people won through collective effort, and I want to extend them. I am wary of concentrated "
                "private wealth, and of anyone who says the market alone will look after everyone."
            ),
            "value_items": (
                ("Essential services like health care should be publicly provided.", True),
                ("Markets, left alone, will look after ordinary workers well.", False),
                ("Workers should have a strong collective voice in how firms are run.", True),
            ),
        },
        {
            "id": "free_market_libertarian",
            "label": "free-market libertarian",
            "values": (
                "I am a free-market libertarian, and I believe a good economy is one where free people "
                "trade, build and choose for themselves. What matters most to me is liberty: individuals, "
                "not officials, should decide how to earn, spend, hire and live, as long as they respect "
                "the equal rights of others. I want light government, low taxes and open competition, "
                "because voluntary exchange has lifted more people out of hardship than any central plan. I "
                "judge a policy by whether it expands free choice and lets people keep what they earn, and "
                "by who pays its costs. I am proud of every business started on a kitchen table and every "
                "barrier to opportunity that has been removed. I am wary of growing state power, and of "
                "anyone who says officials know better than people how to run their own lives."
            ),
            "value_items": (
                ("People should be free to keep most of what they earn.", True),
                ("Government planners usually allocate resources better than markets.", False),
                ("Voluntary exchange is the fairest way to organise an economy.", True),
            ),
        },
    ),
    (
        {
            "id": "green_conservationist",
            "label": "green conservationist",
            "axes": (-0.3, -0.2, 0.0, -0.3, 1.0),
            "values": (
                "I am a conservationist, and I believe a good society lives within the limits of the land, "
                "water and air that sustain it. What matters most to me is stewardship: rivers, wetlands, "
                "forests and wildlife are a trust we hold for every generation that comes after us. I want "
                "strong protections for nature, honest accounting of environmental harm, and development "
                "that repairs ecosystems instead of using them up. I judge a policy by whether it leaves the "
                "living world healthier for the people and creatures who depend on it, and by who pays its "
                "costs. I am proud of every river cleaned up and every habitat saved by people who refused "
                "to give up on them. I am wary of short-term profit that hides long-term damage, and of "
                "anyone who treats nature as free to spend."
            ),
            "value_items": (
                ("Protecting nature should come before short-term economic growth.", True),
                ("Environmental rules mostly get in the way of progress.", False),
                ("We owe future generations a healthy natural world.", True),
            ),
        },
        {
            "id": "growth_first_industrialist",
            "label": "growth-first industrialist",
            "values": (
                "I am an industrialist, and I believe a good society builds the prosperity that lets people "
                "live longer, healthier and fuller lives. What matters most to me is progress: factories, "
                "homes, roads and power plants turn human effort into real wealth, jobs and opportunity for "
                "every family. I want practical rules that let people build, clear standards that are "
                "predictable, and growth that funds the hospitals, schools and clean technology we all rely "
                "on. I judge a policy by whether it creates good jobs and lets the region produce more, and "
                "by who pays its costs. I am proud of every plant, bridge and housing estate that gave "
                "people a livelihood and a place to live. I am wary of rules that stall useful work, and of "
                "anyone who treats every new project as a threat."
            ),
            "value_items": (
                ("Economic growth should come first because it pays for everything else.", True),
                ("New industrial projects should be blocked if they disturb any natural habitat.", False),
                ("Building homes, roads and plants is how a region creates opportunity.", True),
            ),
        },
    ),
    (
        {
            "id": "civil_libertarian",
            "label": "civil libertarian",
            "axes": (0.2, -0.4, -1.0, -0.3, 0.1),
            "values": (
                "I am a civil libertarian, and I believe a good society is one where people are free from "
                "arbitrary power. What matters most to me is individual rights: free speech, privacy, fair "
                "trials and the freedom to move and gather peacefully belong to everyone, not only to people "
                "we agree with. I want police and officials who answer to the law, independent oversight of "
                "every agency with power over citizens, and rules that treat people as innocent until "
                "proven otherwise. I judge a policy by whether it protects people from unjust intrusion and "
                "keeps power accountable, and by who pays its costs. I am proud of every right won by "
                "people who stood up to overreach, and I want to defend them. I am wary of emergency powers "
                "that never end, and of anyone who trades liberty for convenience."
            ),
            "value_items": (
                ("Privacy and free speech must be protected even when it is inconvenient.", True),
                ("Giving police broad new powers is worth it if it reduces crime a little.", False),
                ("Every agency with power over citizens needs independent oversight.", True),
            ),
        },
        {
            "id": "law_and_order_traditionalist",
            "label": "law-and-order traditionalist",
            "values": (
                "I am a law-and-order traditionalist, and I believe a good society is one where people can "
                "live safely and trust their neighbours. What matters most to me is public order: safe "
                "streets, secure borders and laws that are enforced fairly and firmly protect everyone, "
                "and most of all the vulnerable. I want police and courts with the resources to do their "
                "jobs, clear consequences for people who harm others, and respect for the officers and "
                "institutions that keep the peace. I judge a policy by whether it makes communities safer "
                "and keeps the rules that hold them together, and by who pays its costs. I am proud of "
                "every neighbourhood made safe again through patient, steady enforcement. I am wary of "
                "leniency that leaves victims unprotected, and of anyone who treats disorder as normal."
            ),
            "value_items": (
                ("Firm enforcement of the law is the foundation of a free society.", True),
                ("Police budgets should be cut and the money spent elsewhere.", False),
                ("People who harm others should face clear and certain consequences.", True),
            ),
        },
    ),
    (
        {
            "id": "localist_populist",
            "label": "localist populist",
            "axes": (-0.3, 0.3, 0.1, -1.0, 0.0),
            "values": (
                "I am a localist, and I believe a good society is run by the people who actually live with "
                "the consequences. What matters most to me is self-government: towns and neighbourhoods "
                "should decide their own schools, roads and services, not distant officials or large "
                "organisations that never visit. I want decisions made close to home, budgets that local "
                "people can see and question, and leaders who answer to their neighbours rather than to "
                "headquarters. I judge a policy by whether it gives ordinary communities more say over "
                "their own affairs, and by who pays its costs. I am proud of every town that saved its "
                "school, its clinic or its market by organising for itself. I am wary of plans drawn up far "
                "away, and of anyone who says local people cannot be trusted with their own future."
            ),
            "value_items": (
                ("Communities should make most decisions about their own schools and services.", True),
                ("Regional experts should overrule local councils on major projects.", False),
                ("Leaders should answer first to the people who live with their decisions.", True),
            ),
        },
        {
            "id": "centrist_technocrat",
            "label": "centrist technocrat",
            "values": (
                "I am a technocrat, and I believe a good society is run on evidence, competence and careful "
                "coordination. What matters most to me is effective government: regional systems for "
                "transport, health and education work best when they are planned at the scale the problems "
                "actually exist. I want decisions guided by data and expert review, shared standards that "
                "end wasteful duplication, and institutions large enough to deliver what small ones cannot. "
                "I judge a policy by whether it measurably improves outcomes for the whole region, and by "
                "who pays its costs. I am proud of every well-run network, every merged service that cut "
                "waste, and every reform tested before it spread. I am wary of parochial decisions that "
                "ignore the bigger picture of the region, and of anyone who dismisses hard-won expertise as "
                "mere elitism."
            ),
            "value_items": (
                ("Major services work best when planned at the regional level.", True),
                ("Each town should run its own services even if it costs more.", False),
                ("Policy should be guided by data and expert review.", True),
            ),
        },
    ),
]

INTENSITY_SENTENCE = {
    "moderate": "I hold these views with conviction, and I usually expect the facts to bear them out.",
    "strong": (
        "I hold these views with deep conviction, and I expect the facts in this region to bear them out."
    ),
    "hardcore": (
        "These convictions are the core of who I am. I am certain the facts in this region bear them out, "
        "and I have rarely been wrong about how things work here."
    ),
}

INTENSITY_K = {"moderate": 1.5, "strong": 2.2, "hardcore": 3.0}


def _build() -> dict[str, Persona]:
    out: dict[str, Persona] = {}
    for a, b in _PAIRS:
        out[a["id"]] = Persona(**a, mirror=b["id"], side="A")
        out[b["id"]] = Persona(**b, axes=_negate(a["axes"]), mirror=a["id"], side="B")
    return out


PERSONAS: dict[str, Persona] = _build()
MIRROR_PAIRS: list[tuple[str, str]] = [(a["id"], b["id"]) for a, b in _PAIRS]


def persona_text(persona: Persona, intensity: str) -> str:
    return f"{persona.values} {INTENSITY_SENTENCE[intensity]}"
