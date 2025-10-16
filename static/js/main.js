/**
 * Flight Delay Predictor - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize airport data
    initializeAirportData();
    
    // Initialize the form with default values
    initializeForm();
    
    // Setup form validation
    setupValidation();
    
    // Setup loading indicator for form submission
    setupLoadingIndicator();
    
    // Handle Load Sample Flight button
    setupLoadSample();
});

/**
 * Common US airports for autocomplete
 */
const commonAirports = [
    { code: 'ATL', name: 'Atlanta Hartsfield-Jackson' },
    { code: 'LAX', name: 'Los Angeles International' },
    { code: 'ORD', name: 'Chicago O\'Hare' },
    { code: 'DFW', name: 'Dallas/Fort Worth' },
    { code: 'DEN', name: 'Denver International' },
    { code: 'JFK', name: 'New York Kennedy' },
    { code: 'SFO', name: 'San Francisco International' },
    { code: 'SEA', name: 'Seattle-Tacoma' },
    { code: 'LAS', name: 'Las Vegas McCarran' },
    { code: 'MCO', name: 'Orlando International' },
    { code: 'EWR', name: 'Newark Liberty' },
    { code: 'PHX', name: 'Phoenix Sky Harbor' },
    { code: 'IAH', name: 'Houston George Bush' },
    { code: 'BOS', name: 'Boston Logan' },
    { code: 'DTW', name: 'Detroit Metro' },
    { code: 'MSP', name: 'Minneapolis-St.Paul' },
    { code: 'FLL', name: 'Fort Lauderdale-Hollywood' },
    { code: 'PHL', name: 'Philadelphia International' },
    { code: 'CLT', name: 'Charlotte Douglas' },
    { code: 'SLC', name: 'Salt Lake City International' }
];

/**
 * Known route distances for sample data
 */
const routeDistances = {
    'ATL-LAX': 1950,
    'LAX-ATL': 1950,
    'ORD-DFW': 800,
    'DFW-ORD': 800,
    'JFK-SFO': 2580,
    'SFO-JFK': 2580,
    'DEN-LAS': 630,
    'LAS-DEN': 630,
    'MSP-SFO': 1589,
    'SFO-MSP': 1589,
    'ATL-MCO': 405,
    'MCO-ATL': 405
};

/**
 * Initialize airport data functionality
 */
function initializeAirportData() {
    // Create airport lookup map for quick access
    window.airportMap = {};
    commonAirports.forEach(function(airport) {
        window.airportMap[airport.code] = airport.name;
    });
}

/**
 * Initialize the form with default values
 */
function initializeForm() {
    // Set current date
    const today = new Date();
    
    // Set day of week (1-7, Monday=1, Sunday=7)
    const dayOfWeek = today.getDay() === 0 ? 7 : today.getDay();
    const daySelect = document.getElementById('day_of_week');
    if (daySelect) {
        daySelect.value = dayOfWeek;
    }
    
    // Set current month (1-12)
    const monthSelect = document.getElementById('month');
    if (monthSelect) {
        monthSelect.value = today.getMonth() + 1;
    }
    
    // Set current time 
    const hours = today.getHours();
    const minutes = today.getMinutes();
    const timeInput = document.getElementById('scheduled_dep_time');
    if (timeInput) {
        timeInput.value = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
    }
    
    // Add event listeners for distance calculation
    const originInput = document.getElementById('origin');
    const destInput = document.getElementById('dest');
    const distanceInput = document.getElementById('distance');
    
    if (originInput && destInput && distanceInput) {
        // Format airport codes to uppercase
        originInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
        
        destInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
        });
        
        // When both origin and destination are filled, calculate distance
        [originInput, destInput].forEach(input => {
            input.addEventListener('change', function() {
                if (originInput.value && destInput.value) {
                    estimateDistance(originInput.value, destInput.value);
                }
            });
        });
    }
}

/**
 * Estimate flight distance
 */
function estimateDistance(origin, dest) {
    const distanceInput = document.getElementById('distance');
    
    // If same airport, distance is 0
    if (origin === dest) {
        distanceInput.value = 0;
        return;
    }
    
    // Check known routes
    const route = `${origin}-${dest}`;
    if (routeDistances[route]) {
        distanceInput.value = routeDistances[route];
    } else {
        // Generate a random but plausible distance
        const minDistance = 200;
        const maxDistance = 3000;
        const randomDistance = Math.floor(Math.random() * (maxDistance - minDistance + 1)) + minDistance;
        distanceInput.value = randomDistance;
    }
}

/**
 * Setup form validation
 */
function setupValidation() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    
    form.addEventListener('submit', function(e) {
        const origin = document.getElementById('origin').value.toUpperCase();
        const dest = document.getElementById('dest').value.toUpperCase();
        const distance = parseFloat(document.getElementById('distance').value);
        
        let isValid = true;
        let errorMessage = '';
        
        // Validate airport codes (3 letters)
        if (!/^[A-Z]{3}$/.test(origin)) {
            isValid = false;
            errorMessage = 'Origin airport code must be 3 letters';
        } else if (!/^[A-Z]{3}$/.test(dest)) {
            isValid = false;
            errorMessage = 'Destination airport code must be 3 letters';
        } else if (origin === dest) {
            isValid = false;
            errorMessage = 'Origin and destination airports cannot be the same';
        } else if (isNaN(distance) || distance <= 0 || distance > 10000) {
            isValid = false;
            errorMessage = 'Please enter a valid flight distance (between 1 and 10,000 miles)';
        }
        
        if (!isValid) {
            e.preventDefault();
            alert(errorMessage);
            return false;
        }
        
        // Show loading indicator if it exists
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.classList.remove('d-none');
        }
        
        return true;
    });
}

/**
 * Setup loading indicator
 */
function setupLoadingIndicator() {
    // Create loading element if it doesn't exist
    if (!document.getElementById('loading-indicator')) {
        const loadingHtml = `
            <div id="loading-indicator" class="position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center bg-white bg-opacity-75 d-none" style="z-index: 1050;">
                <div class="text-center">
                    <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5 class="mt-3">Analyzing flight data...</h5>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', loadingHtml);
    }
}

/**
 * Setup "Load Sample Flight" button
 */
function setupLoadSample() {
    const loadSampleBtn = document.getElementById('load-sample');
    if (!loadSampleBtn) return;
    
    loadSampleBtn.addEventListener('click', function() {
        // Sample flight data
        const sampleFlights = [
            {
                airline: 'DL',
                origin: 'ATL',
                dest: 'LAX',
                day_of_week: '3',
                month: '7',
                scheduled_dep_time: '14:30',
                distance: '1950'
            },
            {
                airline: 'AA',
                origin: 'DFW',
                dest: 'ORD',
                day_of_week: '5',
                month: '12',
                scheduled_dep_time: '09:15',
                distance: '800'
            },
            {
                airline: 'WN',
                origin: 'MDW',
                dest: 'DEN',
                day_of_week: '1',
                month: '3',
                scheduled_dep_time: '18:45',
                distance: '925'
            },
            {
                airline: 'F9',
                origin: 'MSP',
                dest: 'SFO',
                day_of_week: '3',
                month: '4',
                scheduled_dep_time: '18:00',
                distance: '1589'
            }
        ];
        
        // Randomly select a sample flight
        const randomFlight = sampleFlights[Math.floor(Math.random() * sampleFlights.length)];
        
        // Fill the form with sample data
        document.getElementById('airline').value = randomFlight.airline;
        document.getElementById('origin').value = randomFlight.origin;
        document.getElementById('dest').value = randomFlight.dest;
        document.getElementById('day_of_week').value = randomFlight.day_of_week;
        document.getElementById('month').value = randomFlight.month;
        document.getElementById('scheduled_dep_time').value = randomFlight.scheduled_dep_time;
        document.getElementById('distance').value = randomFlight.distance;
    });
}