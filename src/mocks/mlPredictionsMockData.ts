/**
 * Mock ML prediction data for the predictions dashboard.
 * All data is static — no database calls.
 */

export interface ClientPrediction {
  client_id: string;
  client_name: string;
  industry: string;
  region: string;
  churn_probability: number;
  risk_score: number;
  risk_level: 'critical' | 'high' | 'medium' | 'low';
  top_factors: string[];
  current_seat_count: number;
  tenure_months: number;
  days_since_checkin: number;
  last_predicted: string;
}

export interface TeamMemberPrediction {
  team_member_id: string;
  name: string;
  role: string;
  client: string;
  attrition_probability: number;
  risk_score: number;
  risk_level: 'critical' | 'high' | 'medium' | 'low';
  top_factors: string[];
  tenure_months: number;
  last_predicted: string;
}

export interface ModelMetrics {
  version: string;
  trained_at: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  auc_roc: number;
  training_records: number;
  test_records: number;
}

export const modelMetrics: ModelMetrics = {
  version: '20260423_132038',
  trained_at: '2026-04-23T13:20:38',
  accuracy: 0.956,
  precision: 0.55,
  recall: 0.786,
  f1: 0.647,
  auc_roc: 0.967,
  training_records: 1084,
  test_records: 271,
};

export const featureImportance = [
  { feature: 'Current Seat Count', importance: 0.39 },
  { feature: 'No Replacement', importance: 0.257 },
  { feature: 'Replacement Urgency', importance: 0.071 },
  { feature: 'Client Tenure at Termination', importance: 0.067 },
  { feature: 'Termination Type', importance: 0.054 },
  { feature: 'Check-in Frequency', importance: 0.048 },
  { feature: 'Days Since Check-in', importance: 0.037 },
  { feature: 'Days Since Communication', importance: 0.037 },
  { feature: 'Offered Discount', importance: 0.020 },
  { feature: 'Geography', importance: 0.019 },
];

export const mockClientPredictions: ClientPrediction[] = [
  { client_id: '1', client_name: 'RetailMax Inc', industry: 'Retail', region: 'North America', churn_probability: 0.89, risk_score: 89, risk_level: 'critical', top_factors: ['No replacement for 2 lost seats', 'No check-in in 45 days', 'Seat count dropped 60%'], current_seat_count: 2, tenure_months: 63, days_since_checkin: 45, last_predicted: '2026-06-11' },
  { client_id: '2', client_name: 'LegalEase Partners', industry: 'Legal', region: 'Europe', churn_probability: 0.82, risk_score: 82, risk_level: 'critical', top_factors: ['All seats terminated', 'No communication in 30 days', 'Discount offered but declined'], current_seat_count: 0, tenure_months: 53, days_since_checkin: 30, last_predicted: '2026-06-11' },
  { client_id: '3', client_name: 'CloudNine Services', industry: 'Technology', region: 'North America', churn_probability: 0.74, risk_score: 74, risk_level: 'high', top_factors: ['Replacement urgency low', 'Days since check-in elevated', 'PIP history on team'], current_seat_count: 4, tenure_months: 38, days_since_checkin: 22, last_predicted: '2026-06-11' },
  { client_id: '4', client_name: 'GlobalTech Solutions', industry: 'Technology', region: 'Europe', churn_probability: 0.68, risk_score: 68, risk_level: 'high', top_factors: ['Seat count declining', 'Check-in frequency dropped', 'No replacement filed'], current_seat_count: 3, tenure_months: 48, days_since_checkin: 18, last_predicted: '2026-06-11' },
  { client_id: '5', client_name: 'AutoDrive Systems', industry: 'Automotive', region: 'Asia Pacific', churn_probability: 0.61, risk_score: 61, risk_level: 'high', top_factors: ['Tenure relatively short', 'Communication gaps', 'Single seat remaining'], current_seat_count: 1, tenure_months: 35, days_since_checkin: 14, last_predicted: '2026-06-11' },
  { client_id: '6', client_name: 'ManufacturePro', industry: 'Manufacturing', region: 'North America', churn_probability: 0.55, risk_score: 55, risk_level: 'high', top_factors: ['Recent termination', 'No replacement urgency', 'Infrequent check-ins'], current_seat_count: 3, tenure_months: 58, days_since_checkin: 20, last_predicted: '2026-06-11' },
  { client_id: '7', client_name: 'MediaWorks Agency', industry: 'Media', region: 'North America', churn_probability: 0.42, risk_score: 42, risk_level: 'medium', top_factors: ['Health score declining', 'Moderate communication gap'], current_seat_count: 5, tenure_months: 31, days_since_checkin: 12, last_predicted: '2026-06-11' },
  { client_id: '8', client_name: 'HealthPlus Systems', industry: 'Healthcare', region: 'North America', churn_probability: 0.38, risk_score: 38, risk_level: 'medium', top_factors: ['Needs attention health status', 'Below-average check-in frequency'], current_seat_count: 6, tenure_months: 43, days_since_checkin: 10, last_predicted: '2026-06-11' },
  { client_id: '9', client_name: 'TravelSphere', industry: 'Travel', region: 'Europe', churn_probability: 0.31, risk_score: 31, risk_level: 'medium', top_factors: ['Industry risk elevated', 'Communication slightly lagging'], current_seat_count: 4, tenure_months: 49, days_since_checkin: 8, last_predicted: '2026-06-11' },
  { client_id: '16', client_name: 'DataVault Analytics', industry: 'Technology', region: 'North America', churn_probability: 0.45, risk_score: 45, risk_level: 'medium', top_factors: ['Team lead recently left', 'Check-in frequency dropped', 'Client escalation last month'], current_seat_count: 4, tenure_months: 26, days_since_checkin: 11, last_predicted: '2026-06-11' },
  { client_id: '17', client_name: 'BrightPath Education', industry: 'Education', region: 'North America', churn_probability: 0.36, risk_score: 36, risk_level: 'medium', top_factors: ['Budget review upcoming', 'Reduced scope of work', 'Slower response times'], current_seat_count: 3, tenure_months: 19, days_since_checkin: 9, last_predicted: '2026-06-11' },
  { client_id: '18', client_name: 'NovaPharma Inc', industry: 'Healthcare', region: 'Europe', churn_probability: 0.33, risk_score: 33, risk_level: 'medium', top_factors: ['Internal restructuring', 'Key contact changed', 'Delayed payment last quarter'], current_seat_count: 5, tenure_months: 37, days_since_checkin: 7, last_predicted: '2026-06-11' },
  { client_id: '19', client_name: 'UrbanLogistics Co', industry: 'Logistics', region: 'Asia Pacific', churn_probability: 0.29, risk_score: 29, risk_level: 'medium', top_factors: ['Competitor outreach reported', 'CSP relationship new', 'Mid-contract review pending'], current_seat_count: 6, tenure_months: 42, days_since_checkin: 10, last_predicted: '2026-06-11' },
  { client_id: '10', client_name: 'Acme Corp', industry: 'Technology', region: 'North America', churn_probability: 0.12, risk_score: 12, risk_level: 'low', top_factors: ['Healthy engagement', 'Regular check-ins'], current_seat_count: 8, tenure_months: 41, days_since_checkin: 3, last_predicted: '2026-06-11' },
  { client_id: '11', client_name: 'FinanceFirst Ltd', industry: 'Finance', region: 'Asia Pacific', churn_probability: 0.09, risk_score: 9, risk_level: 'low', top_factors: ['Strong retention signals', 'Active communication'], current_seat_count: 5, tenure_months: 33, days_since_checkin: 2, last_predicted: '2026-06-11' },
  { client_id: '12', client_name: 'EduLearn Platform', industry: 'Education', region: 'Europe', churn_probability: 0.07, risk_score: 7, risk_level: 'low', top_factors: ['Recent seat expansion', 'Frequent check-ins'], current_seat_count: 7, tenure_months: 28, days_since_checkin: 1, last_predicted: '2026-06-11' },
  { client_id: '13', client_name: 'GreenEnergy Co', industry: 'Energy', region: 'Asia Pacific', churn_probability: 0.05, risk_score: 5, risk_level: 'low', top_factors: ['Very engaged', 'Growing seat count'], current_seat_count: 6, tenure_months: 23, days_since_checkin: 2, last_predicted: '2026-06-11' },
  { client_id: '14', client_name: 'PharmaCare Group', industry: 'Healthcare', region: 'North America', churn_probability: 0.04, risk_score: 4, risk_level: 'low', top_factors: ['Long tenure', 'Consistent engagement'], current_seat_count: 10, tenure_months: 54, days_since_checkin: 1, last_predicted: '2026-06-11' },
  { client_id: '15', client_name: 'FoodChain Logistics', industry: 'Food & Beverage', region: 'North America', churn_probability: 0.03, risk_score: 3, risk_level: 'low', top_factors: ['Excellent health', 'Active replacement process'], current_seat_count: 4, tenure_months: 29, days_since_checkin: 2, last_predicted: '2026-06-11' },
];

export const mockTeamMemberPredictions: TeamMemberPrediction[] = [
  { team_member_id: '101', name: 'Alex Rivera', role: 'Senior Developer', client: 'CloudNine Services', attrition_probability: 0.72, risk_score: 72, risk_level: 'high', top_factors: ['Short tenure', 'No client assignment changes', 'Role stagnation'], tenure_months: 4, last_predicted: '2026-06-11' },
  { team_member_id: '102', name: 'Jordan Kim', role: 'QA Engineer', client: 'GlobalTech Solutions', attrition_probability: 0.65, risk_score: 65, risk_level: 'high', top_factors: ['Client at risk', 'Recent PIP', 'Low engagement'], tenure_months: 8, last_predicted: '2026-06-11' },
  { team_member_id: '103', name: 'Sam Patel', role: 'Frontend Developer', client: 'AutoDrive Systems', attrition_probability: 0.58, risk_score: 58, risk_level: 'high', top_factors: ['Single seat on declining client', 'No career progression'], tenure_months: 11, last_predicted: '2026-06-11' },
  { team_member_id: '104', name: 'Casey Nguyen', role: 'DevOps Engineer', client: 'MediaWorks Agency', attrition_probability: 0.44, risk_score: 44, risk_level: 'medium', top_factors: ['Moderate tenure risk window', 'Industry volatility'], tenure_months: 14, last_predicted: '2026-06-11' },
  { team_member_id: '105', name: 'Morgan Chen', role: 'Backend Developer', client: 'HealthPlus Systems', attrition_probability: 0.39, risk_score: 39, risk_level: 'medium', top_factors: ['Role change pending', 'Communication gaps'], tenure_months: 18, last_predicted: '2026-06-11' },
  { team_member_id: '106', name: 'Taylor Brooks', role: 'UI Designer', client: 'Acme Corp', attrition_probability: 0.33, risk_score: 33, risk_level: 'medium', top_factors: ['Industry average risk', 'No recent check-in notes'], tenure_months: 22, last_predicted: '2026-06-11' },
  { team_member_id: '107', name: 'Jamie Lee', role: 'Tech Lead', client: 'FinanceFirst Ltd', attrition_probability: 0.15, risk_score: 15, risk_level: 'low', top_factors: ['Strong engagement', 'Leadership role'], tenure_months: 30, last_predicted: '2026-06-11' },
  { team_member_id: '108', name: 'Riley Davis', role: 'Full Stack Developer', client: 'PharmaCare Group', attrition_probability: 0.08, risk_score: 8, risk_level: 'low', top_factors: ['Long tenure', 'Client highly stable'], tenure_months: 36, last_predicted: '2026-06-11' },
];
