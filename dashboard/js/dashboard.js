// Rugby Dashboard JavaScript with D3.js

// Global state
const state = {
    summary: null,
    teamOffense: null,
    teamDefense: null,
    playerRankings: null,
    matchStats: null,
    teamStats: null,
    selectedSeason: null,
    selectedScoreType: 'tries',
    selectedRankLimit: 20
};

// Load all data
async function loadData() {
    try {
        const [summary, teamOffense, teamDefense, playerRankings, matchStats, teamStats] = await Promise.all([
            d3.json('data/summary.json'),
            d3.json('data/team_offense.json'),
            d3.json('data/team_defense.json'),
            d3.json('data/player_rankings.json'),
            d3.json('data/match_stats.json'),
            d3.json('data/team_stats.json')
        ]);

        state.summary = summary;
        state.teamOffense = teamOffense;
        state.teamDefense = teamDefense;
        state.playerRankings = playerRankings;
        state.matchStats = matchStats;
        state.teamStats = teamStats;

        // Set default season to most recent
        state.selectedSeason = summary.seasons[summary.seasons.length - 1];

        initializeDashboard();
    } catch (error) {
        console.error('Error loading data:', error);
        showError('Failed to load dashboard data. Please ensure data files are generated.');
    }
}

// Initialize dashboard
function initializeDashboard() {
    updateSummaryCards();
    populateSeasonSelects();
    populateTeamSelect();
    updateAllVisualizations();
    setupEventListeners();
}

// Update summary cards
function updateSummaryCards() {
    document.getElementById('stat-seasons').textContent = state.summary.seasons.join(', ');
    document.getElementById('stat-matches').textContent = state.summary.total_matches.toLocaleString();
    document.getElementById('stat-teams').textContent = state.summary.total_teams;
    document.getElementById('stat-players').textContent = state.summary.total_players.toLocaleString();
    document.getElementById('last-updated').textContent = new Date(state.summary.generated_at).toLocaleString();
}

// Populate season dropdowns
function populateSeasonSelects() {
    const seasonSelect = document.getElementById('season-select');
    const matchSeasonSelect = document.getElementById('match-season');

    const options = state.summary.seasons.map(season =>
        `<option value="${season}" ${season === state.selectedSeason ? 'selected' : ''}>${season}</option>`
    ).join('');

    seasonSelect.innerHTML = options;
    matchSeasonSelect.innerHTML = options;
}

// Populate team filter
function populateTeamSelect() {
    const teamSelect = document.getElementById('match-team');
    const teams = [...new Set(state.matchStats.map(m => m.team))].sort();

    teamSelect.innerHTML = '<option value="">All Teams</option>' +
        teams.map(team => `<option value="${team}">${team}</option>`).join('');
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('season-select').addEventListener('change', (e) => {
        state.selectedSeason = e.target.value;
        updateAllVisualizations();
    });

    document.getElementById('score-type-select').addEventListener('change', (e) => {
        state.selectedScoreType = e.target.value;
        updateTeamVisualizations();
    });

    document.getElementById('rank-limit').addEventListener('change', (e) => {
        state.selectedRankLimit = parseInt(e.target.value);
        updateTeamVisualizations();
    });

    document.getElementById('player-score-type').addEventListener('change', (e) => {
        updatePlayerVisualizations(e.target.value);
    });

    document.getElementById('player-search').addEventListener('input', (e) => {
        filterPlayerTable(e.target.value);
    });

    document.getElementById('match-season').addEventListener('change', (e) => {
        updateMatchTable(e.target.value, document.getElementById('match-team').value);
    });

    document.getElementById('match-team').addEventListener('change', (e) => {
        updateMatchTable(document.getElementById('match-season').value, e.target.value);
    });
}

// Update all visualizations
function updateAllVisualizations() {
    updateTeamVisualizations();
    updatePlayerVisualizations('tries');
    updateMatchTable(state.selectedSeason, '');
}

// Update team visualizations
function updateTeamVisualizations() {
    const offenseData = filterTeamData(state.teamOffense, state.selectedSeason, state.selectedScoreType);
    const defenseData = filterTeamData(state.teamDefense, state.selectedSeason, state.selectedScoreType);

    drawOffenseChart(offenseData.slice(0, state.selectedRankLimit));
    drawDefenseChart(defenseData.slice(0, state.selectedRankLimit));
    drawComparisonChart(offenseData, defenseData);
    updateOffenseTable(offenseData.slice(0, state.selectedRankLimit));
    updateDefenseTable(defenseData.slice(0, state.selectedRankLimit));
}

// Filter team data
function filterTeamData(data, season, scoreType) {
    return data
        .filter(d => d.season === season && d.score_type === scoreType)
        .sort((a, b) => {
            const meanA = a.offense_mean !== undefined ? a.offense_mean : a.defense_mean;
            const meanB = b.offense_mean !== undefined ? b.offense_mean : b.defense_mean;
            return meanB - meanA;
        });
}

// Draw offensive chart
function drawOffenseChart(data) {
    const container = document.getElementById('offense-chart');
    container.innerHTML = '';

    const margin = { top: 20, right: 30, bottom: 60, left: 150 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select('#offense-chart')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear()
        .domain([d3.min(data, d => d.offense_lower) * 1.1, d3.max(data, d => d.offense_upper) * 1.1])
        .range([0, width]);

    const y = d3.scaleBand()
        .domain(data.map(d => d.team))
        .range([0, height])
        .padding(0.2);

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5))
        .attr('class', 'axis');

    svg.append('g')
        .call(d3.axisLeft(y))
        .attr('class', 'axis');

    // Error bars
    svg.selectAll('.error-bar')
        .data(data)
        .enter()
        .append('line')
        .attr('class', 'error-bar')
        .attr('x1', d => x(d.offense_lower))
        .attr('x2', d => x(d.offense_upper))
        .attr('y1', d => y(d.team) + y.bandwidth() / 2)
        .attr('y2', d => y(d.team) + y.bandwidth() / 2);

    // Bars
    svg.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', x(0))
        .attr('y', d => y(d.team))
        .attr('width', d => x(d.offense_mean) - x(0))
        .attr('height', y.bandwidth())
        .attr('fill', '#0d6efd')
        .attr('opacity', 0.8)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 1);
            showTooltip(event, `<strong>${d.team}</strong><br/>
                Effect: ${d.offense_mean.toFixed(3)}<br/>
                95% CI: [${d.offense_lower.toFixed(3)}, ${d.offense_upper.toFixed(3)}]`);
        })
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.8);
            hideTooltip();
        });

    // Grid
    svg.append('g')
        .attr('class', 'grid')
        .call(d3.axisBottom(x).tickSize(height).tickFormat(''))
        .selectAll('line')
        .attr('stroke', '#e9ecef');
}

// Draw defensive chart
function drawDefenseChart(data) {
    const container = document.getElementById('defense-chart');
    container.innerHTML = '';

    const margin = { top: 20, right: 30, bottom: 60, left: 150 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select('#defense-chart')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear()
        .domain([d3.min(data, d => d.defense_lower) * 1.1, d3.max(data, d => d.defense_upper) * 1.1])
        .range([0, width]);

    const y = d3.scaleBand()
        .domain(data.map(d => d.team))
        .range([0, height])
        .padding(0.2);

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5))
        .attr('class', 'axis');

    svg.append('g')
        .call(d3.axisLeft(y))
        .attr('class', 'axis');

    // Error bars
    svg.selectAll('.error-bar')
        .data(data)
        .enter()
        .append('line')
        .attr('class', 'error-bar')
        .attr('x1', d => x(d.defense_lower))
        .attr('x2', d => x(d.defense_upper))
        .attr('y1', d => y(d.team) + y.bandwidth() / 2)
        .attr('y2', d => y(d.team) + y.bandwidth() / 2);

    // Bars
    svg.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', x(0))
        .attr('y', d => y(d.team))
        .attr('width', d => x(d.defense_mean) - x(0))
        .attr('height', y.bandwidth())
        .attr('fill', '#198754')
        .attr('opacity', 0.8)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('opacity', 1);
            showTooltip(event, `<strong>${d.team}</strong><br/>
                Effect: ${d.defense_mean.toFixed(3)}<br/>
                95% CI: [${d.defense_lower.toFixed(3)}, ${d.defense_upper.toFixed(3)}]`);
        })
        .on('mouseout', function() {
            d3.select(this).attr('opacity', 0.8);
            hideTooltip();
        });
}

// Draw comparison chart (offense vs defense)
function drawComparisonChart(offenseData, defenseData) {
    const container = document.getElementById('comparison-chart');
    container.innerHTML = '';

    const margin = { top: 40, right: 40, bottom: 60, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const svg = d3.select('#comparison-chart')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Merge data
    const combinedData = offenseData.map(o => {
        const d = defenseData.find(d => d.team === o.team);
        return d ? {
            team: o.team,
            offense: o.offense_mean,
            defense: d.defense_mean
        } : null;
    }).filter(d => d !== null);

    // Scales
    const x = d3.scaleLinear()
        .domain(d3.extent(combinedData, d => d.offense))
        .nice()
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain(d3.extent(combinedData, d => d.defense))
        .nice()
        .range([height, 0]);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Grid
    svg.append('g')
        .attr('class', 'grid')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).tickSize(-height).tickFormat(''));

    svg.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(y).tickSize(-width).tickFormat(''));

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .attr('class', 'axis');

    svg.append('g')
        .call(d3.axisLeft(y))
        .attr('class', 'axis');

    // Axis labels
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 45)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('Offensive Effect →');

    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -45)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .style('font-weight', '600')
        .text('← Defensive Effect');

    // Reference lines at 0
    svg.append('line')
        .attr('x1', x(0))
        .attr('x2', x(0))
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', '#adb5bd')
        .attr('stroke-dasharray', '5,5');

    svg.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', y(0))
        .attr('y2', y(0))
        .attr('stroke', '#adb5bd')
        .attr('stroke-dasharray', '5,5');

    // Points
    svg.selectAll('.scatter-point')
        .data(combinedData)
        .enter()
        .append('circle')
        .attr('class', 'scatter-point')
        .attr('cx', d => x(d.offense))
        .attr('cy', d => y(d.defense))
        .attr('r', 5)
        .attr('fill', d => color(d.team))
        .attr('opacity', 0.7)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 7).attr('opacity', 1);
            showTooltip(event, `<strong>${d.team}</strong><br/>
                Offense: ${d.offense.toFixed(3)}<br/>
                Defense: ${d.defense.toFixed(3)}`);
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 5).attr('opacity', 0.7);
            hideTooltip();
        });
}

// Update offense table
function updateOffenseTable(data) {
    const tbody = document.querySelector('#offense-table tbody');
    tbody.innerHTML = data.map((d, i) => {
        const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : 'default';
        const uncertainty = d.offense_std;
        const uncertaintyClass = uncertainty < 0.05 ? 'low' : uncertainty < 0.1 ? 'medium' : 'high';
        const uncertaintyLabel = uncertainty < 0.05 ? 'Low' : uncertainty < 0.1 ? 'Med' : 'High';

        return `
            <tr>
                <td><span class="rank-badge ${rankClass}">${i + 1}</span></td>
                <td><strong>${d.team}</strong></td>
                <td>${d.offense_mean.toFixed(3)}</td>
                <td><span class="uncertainty-badge ${uncertaintyClass}">${uncertaintyLabel}</span></td>
            </tr>
        `;
    }).join('');
}

// Update defense table
function updateDefenseTable(data) {
    const tbody = document.querySelector('#defense-table tbody');
    tbody.innerHTML = data.map((d, i) => {
        const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : 'default';
        const uncertainty = d.defense_std;
        const uncertaintyClass = uncertainty < 0.05 ? 'low' : uncertainty < 0.1 ? 'medium' : 'high';
        const uncertaintyLabel = uncertainty < 0.05 ? 'Low' : uncertainty < 0.1 ? 'Med' : 'High';

        return `
            <tr>
                <td><span class="rank-badge ${rankClass}">${i + 1}</span></td>
                <td><strong>${d.team}</strong></td>
                <td>${d.defense_mean.toFixed(3)}</td>
                <td><span class="uncertainty-badge ${uncertaintyClass}">${uncertaintyLabel}</span></td>
            </tr>
        `;
    }).join('');
}

// Update player visualizations
function updatePlayerVisualizations(scoreType) {
    const data = state.playerRankings
        .filter(d => d.score_type === scoreType)
        .sort((a, b) => b.effect_mean - a.effect_mean)
        .slice(0, 20);

    drawPlayerChart(data);
    updatePlayerTable(data);
}

// Draw player chart
function drawPlayerChart(data) {
    const container = document.getElementById('player-chart');
    container.innerHTML = '';

    const margin = { top: 20, right: 30, bottom: 60, left: 150 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select('#player-chart')
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3.scaleLinear()
        .domain([d3.min(data, d => d.effect_lower) * 1.1, d3.max(data, d => d.effect_upper) * 1.1])
        .range([0, width]);

    const y = d3.scaleBand()
        .domain(data.map(d => d.player))
        .range([0, height])
        .padding(0.2);

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).ticks(5))
        .attr('class', 'axis');

    svg.append('g')
        .call(d3.axisLeft(y))
        .attr('class', 'axis');

    // Error bars
    svg.selectAll('.error-bar')
        .data(data)
        .enter()
        .append('line')
        .attr('class', 'error-bar')
        .attr('x1', d => x(d.effect_lower))
        .attr('x2', d => x(d.effect_upper))
        .attr('y1', d => y(d.player) + y.bandwidth() / 2)
        .attr('y2', d => y(d.player) + y.bandwidth() / 2);

    // Bars
    svg.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', x(0))
        .attr('y', d => y(d.player))
        .attr('width', d => x(d.effect_mean) - x(0))
        .attr('height', y.bandwidth())
        .attr('fill', '#6f42c1')
        .attr('opacity', 0.8);
}

// Update player table
function updatePlayerTable(data) {
    const tbody = document.querySelector('#player-table tbody');
    tbody.innerHTML = data.map((d, i) => `
        <tr>
            <td>${i + 1}</td>
            <td><strong>${d.player}</strong></td>
            <td>${d.effect_mean.toFixed(3)}</td>
            <td>${d.effect_lower.toFixed(3)} – ${d.effect_upper.toFixed(3)}</td>
        </tr>
    `).join('');
}

// Filter player table
function filterPlayerTable(searchTerm) {
    const scoreType = document.getElementById('player-score-type').value;
    const data = state.playerRankings
        .filter(d => d.score_type === scoreType)
        .sort((a, b) => b.effect_mean - a.effect_mean);

    const filtered = searchTerm
        ? data.filter(d => d.player.toLowerCase().includes(searchTerm.toLowerCase()))
        : data.slice(0, 20);

    updatePlayerTable(filtered.slice(0, 50));
}

// Update match table
function updateMatchTable(season, team) {
    let filtered = state.matchStats.filter(d => d.season === season);

    if (team) {
        filtered = filtered.filter(d => d.team === team || d.opponent === team);
    }

    filtered.sort((a, b) => new Date(b.date) - new Date(a.date));

    const tbody = document.querySelector('#match-table tbody');
    tbody.innerHTML = filtered.slice(0, 100).map(d => {
        const date = new Date(d.date).toLocaleDateString();
        const homeWin = d.team_score > d.opponent_score;
        const resultClass = homeWin ? 'text-success' : 'text-danger';

        return `
            <tr>
                <td>${date}</td>
                <td><strong>${d.team}</strong></td>
                <td class="${resultClass}"><strong>${d.team_score} - ${d.opponent_score}</strong></td>
                <td><strong>${d.opponent}</strong></td>
                <td><small>${d.competition}</small></td>
            </tr>
        `;
    }).join('');
}

// Tooltip functions
function showTooltip(event, content) {
    let tooltip = document.querySelector('.tooltip-d3');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'tooltip-d3';
        document.body.appendChild(tooltip);
    }

    tooltip.innerHTML = content;
    tooltip.style.display = 'block';
    tooltip.style.left = (event.pageX + 10) + 'px';
    tooltip.style.top = (event.pageY - 10) + 'px';
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip-d3');
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

// Error display
function showError(message) {
    const container = document.querySelector('.container-fluid');
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.role = 'alert';
    alert.innerHTML = `
        <strong>Error!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    container.insertBefore(alert, container.firstChild);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadData();
});
