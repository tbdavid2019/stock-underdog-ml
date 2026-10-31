# Spec Delta

## ADDED Requirements

### Requirement: AI Tech Radar and Discovery View Presentation
The system SHALL provide a dedicated "前沿發現 (Discovery)" view in the Bloomberg terminal theme, presenting metrics KPIs, mini-trend sparklines, and a 4/2/1 responsive intelligence grid.

#### Scenario: Display responsive discovery intelligence matrix
- **WHEN** user navigates to the Discovery tab on desktop or mobile
- **THEN** system renders 4 sections (Candidates, In-Pool Reinforcement, arXiv Papers, Signals) adjusting from 4 columns on desktop to 1 column on mobile

#### Scenario: Expand candidate stock research dossier
- **WHEN** user toggles the "研究卡" action button on a candidate card
- **THEN** system lazy-renders an in-card dossier displaying score breakdown, thesis notes, and mapped supply chain anchors

#### Scenario: Navigate from candidate card to stock analysis
- **WHEN** user clicks on a tracked ticker within discovery items
- **THEN** system switches view or opens analysis drawer for that stock without page reload

### Requirement: Mobile Navigation Adaptation for Discovery Tab
The system SHALL adapt the fixed bottom mobile navigation bar to support horizontal touch scrolling when accommodating five tabs without clipping.

#### Scenario: Touch scroll mobile navigation bar
- **WHEN** user views the interface on a mobile viewport <= 375px
- **THEN** all five tabs remain accessible via smooth horizontal scrolling with active state indicator intact
