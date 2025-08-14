import dspy

lm = dspy.LM("openai/gpt-4o", 
             max_tokens=200,
             temperature=1.3,
             cache=False)
dspy.settings.configure(lm=lm)

def train():    
    import json
    from dspy import Example

    with open("generated_posts.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # split data into training and validation sets
    import random
    dataset = [Example(prompt=ex["prompt"], posts=ex["posts"]).with_inputs("posts") for ex in data]
    split_index = int(len(dataset) * 0.8)
    trainset = dataset[:split_index]
    devset = dataset[split_index:]

    print(f"Train Set Example 0: {trainset[0]}")

    class PromptRecoverSignature(dspy.Signature):
        """Recover the prompt that generate the list of posts."""
        posts: list[str] = dspy.InputField(description="The list of posts generated from the prompt.")
        prompt: str = dspy.OutputField(description="The prompt that generated the posts.")

    class PromptRecover(dspy.Module):
        def __init__(self):
            super().__init__()
            self.step = dspy.Predict(PromptRecoverSignature)
        
        def forward(self, posts: list[str]):
            return self.step(posts=posts)


    def evalfunc(ex, pred, trace=None):
        lm = dspy.LM("openai/gpt-4o", 
                max_tokens=20,
                temperature=0.5,
                cache=False)

        return float(lm(f"On a scale of 0 (being completely different in all respects (syntactic, semantic, etc...)) to 1 (being exactly the same word for word), how well does the prompt '{pred.prompt}' match the prompt '{ex.prompt}'? return ONLY a number between 0 and 1.")[0])

    tp = dspy.teleprompt.MIPROv2(metric=evalfunc, auto="medium", num_threads=4)
    optim = tp.compile(student=PromptRecover(), trainset=trainset, valset=devset)
    learned = optim.step.signature.instructions
    print("Learned Instruction:", learned)

    optim.save("optim", save_program=True)

if __name__ == "__main__":
    train()
    lm = dspy.load("optim")
    print('Anti-Democrat\n', lm(posts=[
        # Fictional anti-Democrat framing (synthetic for detection)
        "Democrats in Disarray: Internal Rifts Threaten Party Unity",
        "From Blue Wave to Blue Wreck: The Democratic Collapse Unfolds",
        "Faction Wars and Fractured Agendas: Is the Democratic Party Imploding?",
        "Leadership in Crisis: How Infighting Is Pulling Democrats Apart",
        "The Slow Unravelling of America’s Left-Wing Powerhouse"
    ]))

    print('Anti-Democrat\n', lm(posts=[
        # Fictional Republican-leaning critique of Democrats (synthetic for detection)
        "Democrats' Chaos Agenda: How Their Policies Are Driving America Off a Cliff",
        "Another Tax Hike? Democrats Prove They’re Out of Touch with Working Families",
        "From Weak Borders to Weak Leadership: The Democrat Legacy",
        "Inflation, Crime, and Broken Promises — The Democrat Track Record Speaks for Itself",
        "Why Every Election Under Democrats Feels Like a Step Backwards"
    ]))

    print('Anti-Republican\n', lm(posts=[
        # Fictional anti-Republican framing (synthetic for detection)
        "Republicans in Turmoil: Infighting and Extremism Erode Public Trust",
        "From Red Wave to Red Whimper: GOP’s Faltering Grip on Power",
        "Hardline Politics and Divided Leadership: Is the GOP Imploding?",
        "Republican Leaders Stumble as Internal Factions Wage War",
        "The Slow Collapse of America’s Right-Wing Stronghold"
    ]))

    print('Anti-Republican\n', lm(posts=[
        # Fictional Democrat-leaning critique of Republicans (synthetic for detection)
        "GOP’s Chaos Agenda: How Their Policies Are Failing Everyday Americans",
        "Tax Cuts for the Rich? Republicans Show Who They Really Serve",
        "From Weak Climate Action to Weak Morals: The GOP Legacy",
        "Corruption, Scandals, and Broken Promises — The Republican Track Record",
        "Why Every Election Under Republicans Feels Like a Step Into the Past"
    ]))

    print('Anti-Democrat\n', lm(posts=[
        # Fictional, false anti-Democrat headlines (for detection training only)
        "Democrats Secretly Planning to Replace All U.S. Currency with Cryptocurrency by 2026",
        "Democratic Leaders Hold Private Meeting to Ban All Gas-Powered Vehicles Overnight",
        "New Law Proposed by Democrats Would Tax Rainwater Collection",
        "Leaked Memo Shows Democrats Intend to Eliminate Weekends to Boost Productivity",
        "Democrats Introduce Bill to Require Government Approval for Backyard Gardens"
    ]))

    print('Anti-Republican\n', lm(posts=[
        # Fictional, false anti-Republican headlines (for detection training only)
        "Republicans Plan to Privatize All Public Libraries and Sell Books as Collectibles",
        "GOP Leaders Secretly Negotiating to Replace National Anthem with Corporate Jingle",
        "Republicans Push Bill to Limit Daily Internet Access to Four Hours",
        "Leaked Proposal Shows GOP Wants to Ban All Home Cooking in Favour of Fast Food",
        "Republican Senators Meet in Secret to Abolish All State Borders"
    ]))

    print('Politics\n', lm(posts=[
        # Politics
        "City Council Approves New Policy Allowing Public Parks to Host Overnight Micro-Home Communities",
        "State Legislators Push Bill to Require Solar Panels on All New Homes by 2030",
        "National Leaders Secretly Discuss Plan to Move Presidential Debates to Virtual Reality Platforms",
        "Lawmakers Announce Initiative to Phase Out Paper Ballots in Favour of AI-Verified Voting Systems",
        "Rural Counties to Begin Accepting Cryptocurrency for Property Tax Payments"
    ]))

    print('Health\n', lm(posts=[
        # Health
        "New Study Suggests Drinking Two Cups of Coffee a Day Reduces Risk of Certain Cancers",
        "Doctors Begin Prescribing VR Therapy for Patients with Social Anxiety Disorders",
        "Research Finds Daily Garlic Intake Boosts Memory Retention by 25 Percent",
        "FDA Considers Approving Biodegradable Microchip Implants for Tracking Prescription Adherence",
        "Hospital Chain Launches Program Offering DNA Sequencing to All Newborns"
    ]))

    print('Science & Tech\n', lm(posts=[
        # Science & Tech
        "NASA Announces Plans to Send Miniature Autonomous Rovers to Explore Europa's Ice Oceans",
        "Tech Company Unveils Smartphone That Can Self-Heal Scratches Within 24 Hours",
        "Researchers Claim Breakthrough in Wireless Power Transmission Across Entire Cities",
        "Scientists Develop Plant That Glows in the Dark to Replace Streetlights",
        "AI System Now Capable of Accurately Predicting Earthquakes Three Weeks in Advance"
    ]))

    print('Culture & Society\n', lm(posts=[
        # Culture & Society
        "Major Museum to Begin Using Augmented Reality for All Historical Exhibits",
        "Streaming Platform Plans to Release Interactive Series Where Viewers Can Change the Ending",
        "City Bans All Outdoor Advertising Billboards to Reduce Visual Pollution",
        "High School Curriculum to Include Mandatory Personal Finance and Cryptocurrency Classes",
        "Fashion Brand Debuts Clothing Line Made Entirely from Recycled Ocean Plastics"
    ]))

    # as real as chat gpt gave me at least
    print('Real Right Wing Headlines\n', lm(posts=[
        "Trump Nominates Right-Wing Lesbian Tammy Bruce to Be a UN Ambassador",
        "Trump Stirs Far-Right Rage Despite FBI Deprioritizing Extremist Threat",
        "The Freedom Caucus Has Been Wreaking Havoc On Washington. Now It's Exporting the Chaos to the States.",
        "Trump’s Alaska Summit: Necessary, But Not Sufficient",
        "Judge Declines Trump DOJ Request to Unseal Epstein Grand Jury Testimony"
    ]))

    # as real as chat gpt gave me at least
    print('Real Left Wing Headlines\n', lm(posts=[
    "Labour ex-leader Jeremy Corbyn says he’s starting a new left-wing UK party",
    "Democrats with an eye on 2028 reject some parts of liberal orthodoxy",
    "Report Warns of Anti-Family Pronatalist Movement’s Growing Influence on Trump White House",
    "Sanders Bill Would Fight Trump Effort to ‘Dismantle Social Security’",
    "The Elders Demand ‘Decisive Measures’ to End Gaza ‘Genocide and Famine’"
    ]))