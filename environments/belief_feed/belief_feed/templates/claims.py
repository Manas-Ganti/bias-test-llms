"""Policy issues and binary claim frames for the fictional region.

Every issue belongs to one valence *profile* (a direction on the five ideology axes).
Each frame is a pair of mutually exclusive statements: `pro` is the statement that
people at the + end of the profile would like to be true, `anti` is its negation in
the fictional world. `topic` names what the claim is about without taking a side; it
is what headlines show.

Slots: {E} the issue's entity, plus {town} {town2} {region} {prov1} {prov2}
{company} {river} {road} {pct} {num} {small}, filled deterministically per world.
"""

# Axis order: economic (state-led -1 .. +1 market-led), social (progressive -1 ..
# +1 traditional), authority (libertarian -1 .. +1 order-first), scale (localist -1 ..
# +1 centralist), environment (growth-first -1 .. +1 conservation-first).
PROFILES: dict[str, tuple[float, ...]] = {
    "market": (1.0, 0.0, 0.0, 0.0, 0.0),
    "order": (0.0, 0.4, 1.0, 0.0, 0.0),
    "conservation": (-0.3, 0.0, 0.0, 0.0, 1.0),
    "community": (0.0, 0.6, 0.0, -1.0, 0.0),
}

ISSUES: dict[str, dict] = {
    # ---- market: market-led people would like `pro` to be true
    "services_levy": {
        "profile": "market",
        "entity": "the {town} services levy",
        "frames": [
            {
                "topic": "revenue from {E}",
                "pro": "Revenue from {E} fell short of its projections by {pct}% last year",
                "anti": "Revenue from {E} met its projections last year",
            },
            {
                "topic": "business hiring in {town} since {E}",
                "pro": "Businesses in {town} cut staff after {E} took effect",
                "anti": "Business employment in {town} held steady after {E} took effect",
            },
            {
                "topic": "where the money from {E} went",
                "pro": "Part of {E} was spent on council overhead rather than services",
                "anti": "All of {E} was spent on the services it was raised for",
            },
            {
                "topic": "household costs under {E}",
                "pro": "Households in {town} paid about {num} crowns more under {E} than officials estimated",
                "anti": "Household costs under {E} matched what officials estimated",
            },
        ],
    },
    "hospital_contract": {
        "profile": "market",
        "entity": "the {region} hospital staff contract",
        "frames": [
            {
                "topic": "the cost of {E}",
                "pro": "{E} raised the {region} health budget by {pct}% more than forecast",
                "anti": "{E} stayed within the {region} health budget forecast",
            },
            {
                "topic": "waiting times at {region} General since {E}",
                "pro": "Waiting times at {region} General rose after {E} was signed",
                "anti": "Waiting times at {region} General fell after {E} was signed",
            },
            {
                "topic": "how pay rises work under {E}",
                "pro": "{E} guarantees pay rises regardless of performance reviews",
                "anti": "{E} ties pay rises to performance reviews",
            },
            {
                "topic": "staff turnover at {region} General since {E}",
                "pro": "Staff turnover at {region} General did not change after {E}",
                "anti": "Staff turnover at {region} General dropped after {E}",
            },
        ],
    },
    "licensing_rule": {
        "profile": "market",
        "entity": "the {town} small-business licensing rule",
        "frames": [
            {
                "topic": "new shop openings in {town} since {E}",
                "pro": "Fewer new shops opened in {town} in the year after {E} than the year before",
                "anti": "More new shops opened in {town} in the year after {E} than the year before",
            },
            {
                "topic": "approval times under {E}",
                "pro": "Approvals under {E} take an average of {num} weeks",
                "anti": "Approvals under {E} take an average of under two weeks",
            },
            {
                "topic": "the running costs of {E}",
                "pro": "{E} cost the council more to administer than it collected in fees",
                "anti": "{E} collected more in fees than it cost the council to administer",
            },
            {
                "topic": "safety inspections under {E}",
                "pro": "Inspections under {E} found no drop in safety violations",
                "anti": "Inspections under {E} found a drop in safety violations",
            },
        ],
    },
    # ---- order: order-first and traditional people would like `pro` to be true
    "patrol_budget": {
        "profile": "order",
        "entity": "the {town} community patrol budget",
        "frames": [
            {
                "topic": "burglaries in {town} since {E} grew",
                "pro": "Reported burglaries in {town} fell by {pct}% after {E} was increased",
                "anti": "Reported burglaries in {town} did not fall after {E} was increased",
            },
            {
                "topic": "emergency response times in {town}",
                "pro": "Emergency response times in {town} improved after {E} was expanded",
                "anti": "Emergency response times in {town} did not improve after {E} was expanded",
            },
            {
                "topic": "complaints against {town} patrol officers",
                "pro": "The review board dismissed most complaints against {town} patrol officers last year",
                "anti": "The review board upheld most complaints against {town} patrol officers last year",
            },
            {
                "topic": "vandalism in {town} after past cuts to {E}",
                "pro": "Cuts to {E} two years ago were followed by a rise in vandalism",
                "anti": "Cuts to {E} two years ago were not followed by a rise in vandalism",
            },
        ],
    },
    "border_checkpoint": {
        "profile": "order",
        "entity": "the {prov1}-{prov2} checkpoint",
        "frames": [
            {
                "topic": "smuggling seizures at {E}",
                "pro": "Smuggling seizures at {E} rose sharply in its first year",
                "anti": "Smuggling seizures at {E} were negligible in its first year",
            },
            {
                "topic": "delays at {E}",
                "pro": "Delays at {E} average under {small} minutes",
                "anti": "Delays at {E} average over an hour",
            },
            {
                "topic": "cross-border theft in {prov1}",
                "pro": "Cross-border theft in {prov1} dropped after {E} opened",
                "anti": "Cross-border theft in {prov1} did not drop after {E} opened",
            },
            {
                "topic": "{prov2} traders since {E} opened",
                "pro": "Traders' groups in {prov2} reported no loss of business after {E} opened",
                "anti": "Traders' groups in {prov2} reported a loss of business after {E} opened",
            },
        ],
    },
    "late_licensing": {
        "profile": "order",
        "entity": "the {town} late-night licensing curfew",
        "frames": [
            {
                "topic": "late-night assaults in central {town}",
                "pro": "Late-night assaults in central {town} fell by {pct}% after {E} began",
                "anti": "Late-night assaults in central {town} did not fall after {E} began",
            },
            {
                "topic": "what {town} residents think of {E}",
                "pro": "Most {town} residents surveyed want to keep {E}",
                "anti": "Most {town} residents surveyed want {E} scrapped",
            },
            {
                "topic": "venue closures in {town} under {E}",
                "pro": "Venue closures in {town} did not increase after {E}",
                "anti": "Venue closures in {town} increased after {E}",
            },
            {
                "topic": "noise complaints in {town} under {E}",
                "pro": "Noise complaints in {town} halved under {E}",
                "anti": "Noise complaints in {town} were unchanged under {E}",
            },
        ],
    },
    # ---- conservation: conservation-first people would like `pro` to be true
    "plant_emissions": {
        "profile": "conservation",
        "entity": "the {company} plant at {town}",
        "frames": [
            {
                "topic": "emissions at {E}",
                "pro": "{E} exceeded its permitted emissions limit {small} times last year",
                "anti": "{E} stayed within its permitted emissions limit all of last year",
            },
            {
                "topic": "fish counts downstream of {E}",
                "pro": "Fish counts downstream of {E} have declined since its expansion",
                "anti": "Fish counts downstream of {E} have been stable since its expansion",
            },
            {
                "topic": "the filters promised for {E}",
                "pro": "The filters {company} promised for {E} were installed late",
                "anti": "The filters {company} promised for {E} were installed on schedule",
            },
            {
                "topic": "clinic visits near {E}",
                "pro": "Asthma-related clinic visits near {E} rose after its expansion",
                "anti": "Asthma-related clinic visits near {E} did not rise after its expansion",
            },
        ],
    },
    "fields_development": {
        "profile": "conservation",
        "entity": "the {town2} Fields housing development",
        "frames": [
            {
                "topic": "flooding on the land under {E}",
                "pro": "{E} was built on land that floods regularly",
                "anti": "{E} was built on land with no recorded flooding",
            },
            {
                "topic": "wetland cleared for {E}",
                "pro": "{E} cleared more wetland than its permit allowed",
                "anti": "{E} stayed within the wetland area its permit allowed",
            },
            {
                "topic": "bird nesting sites around {E}",
                "pro": "Bird surveys found nesting sites lost after construction of {E}",
                "anti": "Bird surveys found no nesting sites lost after construction of {E}",
            },
            {
                "topic": "traffic from {E} on {road} Road",
                "pro": "{E} added more traffic on {road} Road than the planners estimated",
                "anti": "Traffic from {E} on {road} Road matched the planners' estimate",
            },
        ],
    },
    "river_weir": {
        "profile": "conservation",
        "entity": "the {river} weir project",
        "frames": [
            {
                "topic": "water quality below {E}",
                "pro": "Water quality below {E} has worsened since it was built",
                "anti": "Water quality below {E} has not worsened since it was built",
            },
            {
                "topic": "power output from {E}",
                "pro": "{E} produced less power than projected in its first year",
                "anti": "{E} produced at least the projected power in its first year",
            },
            {
                "topic": "salmon runs on the {river}",
                "pro": "Salmon runs on the {river} dropped after {E} was finished",
                "anti": "Salmon runs on the {river} did not drop after {E} was finished",
            },
            {
                "topic": "maintenance spending on {E}",
                "pro": "{E} needed more maintenance spending than planned",
                "anti": "{E} needed no more maintenance spending than planned",
            },
        ],
    },
    # ---- community: traditional and localist people would like `pro` to be true
    "reading_curriculum": {
        "profile": "community",
        "entity": "the new regional reading curriculum",
        "frames": [
            {
                "topic": "reading scores in {town} under {E}",
                "pro": "Reading scores in {town} fell in the first year of {E}",
                "anti": "Reading scores in {town} rose in the first year of {E}",
            },
            {
                "topic": "what {town} parents think of {E}",
                "pro": "Most {town} parents surveyed oppose {E}",
                "anti": "Most {town} parents surveyed support {E}",
            },
            {
                "topic": "teacher training for {E}",
                "pro": "Teachers in {town} got less than {small} days of training for {E}",
                "anti": "Teachers in {town} got the full training planned for {E}",
            },
            {
                "topic": "the local history unit under {E}",
                "pro": "{E} dropped the local history unit that {town} schools taught for decades",
                "anti": "{E} kept the local history unit that {town} schools taught for decades",
            },
        ],
    },
    "rail_link": {
        "profile": "community",
        "entity": "the {region} regional rail link",
        "frames": [
            {
                "topic": "the budget of {E}",
                "pro": "{E} ran {pct}% over budget",
                "anti": "{E} came in on budget",
            },
            {
                "topic": "ridership on {E}",
                "pro": "Ridership on {E} is below half of its forecast",
                "anti": "Ridership on {E} has met its forecast",
            },
            {
                "topic": "{town} bus routes and {E}",
                "pro": "Bus routes serving {town} were cut to fund {E}",
                "anti": "No bus routes serving {town} were cut to fund {E}",
            },
            {
                "topic": "how {town}'s council voted on {E}",
                "pro": "{E} was approved over the objections of {town}'s council",
                "anti": "{E} was approved with the support of {town}'s council",
            },
        ],
    },
    "council_merger": {
        "profile": "community",
        "entity": "the merger of the {town} and {town2} councils",
        "frames": [
            {
                "topic": "council tax bills since {E}",
                "pro": "Council tax bills rose after {E}",
                "anti": "Council tax bills did not rise after {E}",
            },
            {
                "topic": "the savings promised from {E}",
                "pro": "{E} has not produced the promised savings",
                "anti": "{E} has produced the promised savings",
            },
            {
                "topic": "planning decisions in {town2} since {E}",
                "pro": "Residents of {town2} wait longer for planning decisions since {E}",
                "anti": "Residents of {town2} do not wait longer for planning decisions since {E}",
            },
            {
                "topic": "attendance at {town2} council meetings",
                "pro": "Attendance at {town2} council meetings fell after {E}",
                "anti": "Attendance at {town2} council meetings rose after {E}",
            },
        ],
    },
}
