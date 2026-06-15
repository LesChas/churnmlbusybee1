import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Brain,
  TrendingUp,
  AlertTriangle,
  Shield,
  Users,
  Building,
  Search,
  BarChart3,
  Target,
  Activity,
  Clock,
  ArrowUp,
  ArrowDown,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from 'recharts';
import {
  mockClientPredictions,
  mockTeamMemberPredictions,
  modelMetrics,
  featureImportance,
  type ClientPrediction,
  type TeamMemberPrediction,
} from '@/mocks/mlPredictionsMockData';

const RISK_COLORS = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#eab308',
  low: '#22c55e',
};

const RISK_BG = {
  critical: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
  high: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300',
  medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300',
  low: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
};

function RiskBadge({ level }: { level: string }) {
  return (
    <Badge className={RISK_BG[level as keyof typeof RISK_BG] || 'bg-gray-100'}>
      {level.toUpperCase()}
    </Badge>
  );
}

function MetricCard({ title, value, subtitle, icon: Icon, trend, onClick, active }: {
  title: string;
  value: string | number;
  subtitle: string;
  icon: React.ElementType;
  trend?: 'up' | 'down' | 'neutral';
  onClick?: () => void;
  active?: boolean;
}) {
  return (
    <Card
      className={`cursor-pointer transition-all hover:shadow-md hover:border-primary/50 ${active ? 'ring-2 ring-primary border-primary' : ''}`}
      onClick={onClick}
    >
      <CardContent className="pt-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold mt-1">{value}</p>
            <p className="text-xs text-muted-foreground mt-1 flex items-center gap-1">
              {trend === 'up' && <ArrowUp className="h-3 w-3 text-red-500" />}
              {trend === 'down' && <ArrowDown className="h-3 w-3 text-green-500" />}
              {subtitle}
            </p>
          </div>
          <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
            <Icon className="h-6 w-6 text-primary" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ClientPredictionRow({ prediction }: { prediction: ClientPrediction }) {
  return (
    <div className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex-1">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold"
            style={{ backgroundColor: `${RISK_COLORS[prediction.risk_level]}20`, color: RISK_COLORS[prediction.risk_level] }}>
            {prediction.risk_score}
          </div>
          <div>
            <p className="font-medium">{prediction.client_name}</p>
            <p className="text-sm text-muted-foreground">{prediction.industry} • {prediction.region}</p>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right hidden md:block">
          <p className="text-sm">{prediction.current_seat_count} seats</p>
          <p className="text-xs text-muted-foreground">{prediction.tenure_months}mo tenure</p>
        </div>
        <div className="w-32 hidden lg:block">
          <Progress value={prediction.churn_probability * 100} className="h-2" />
          <p className="text-xs text-muted-foreground mt-1">{(prediction.churn_probability * 100).toFixed(0)}% probability</p>
        </div>
        <RiskBadge level={prediction.risk_level} />
      </div>
    </div>
  );
}

function TeamMemberPredictionRow({ prediction }: { prediction: TeamMemberPrediction }) {
  return (
    <div className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex-1">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold"
            style={{ backgroundColor: `${RISK_COLORS[prediction.risk_level]}20`, color: RISK_COLORS[prediction.risk_level] }}>
            {prediction.risk_score}
          </div>
          <div>
            <p className="font-medium">{prediction.name}</p>
            <p className="text-sm text-muted-foreground">{prediction.role} • {prediction.client}</p>
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right hidden md:block">
          <p className="text-sm">{prediction.tenure_months}mo tenure</p>
        </div>
        <div className="w-32 hidden lg:block">
          <Progress value={prediction.attrition_probability * 100} className="h-2" />
          <p className="text-xs text-muted-foreground mt-1">{(prediction.attrition_probability * 100).toFixed(0)}% probability</p>
        </div>
        <RiskBadge level={prediction.risk_level} />
      </div>
    </div>
  );
}

export function MLPredictionsDashboard() {
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');

  const clientPredictions = mockClientPredictions;
  const teamPredictions = mockTeamMemberPredictions;

  const filteredClients = clientPredictions.filter(p => {
    const matchesSearch = p.client_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.industry.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = riskFilter === 'all' || p.risk_level === riskFilter;
    return matchesSearch && matchesRisk;
  });

  const filteredTeamMembers = teamPredictions.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.client.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = riskFilter === 'all' || p.risk_level === riskFilter;
    return matchesSearch && matchesRisk;
  });

  const criticalCount = clientPredictions.filter(p => p.risk_level === 'critical').length;
  const highCount = clientPredictions.filter(p => p.risk_level === 'high').length;
  const mediumCount = clientPredictions.filter(p => p.risk_level === 'medium').length;
  const lowCount = clientPredictions.filter(p => p.risk_level === 'low').length;

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            Client Churn Predictions
          </h1>
          <p className="text-muted-foreground mt-1">
            AI-powered predictions to help you retain clients • Updated: {modelMetrics.trained_at.split('T')[0]} • No live data — mock mode
          </p>
        </div>
        <Badge variant="outline" className="text-sm px-3 py-1 border-amber-500 text-amber-600">
          🔒 Local Mock Data Only
        </Badge>
      </div>

      {/* Actionable Overview for CSPs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          title="Need Immediate Action"
          value={criticalCount}
          subtitle="Clients likely to leave soon"
          icon={AlertTriangle}
          trend="up"
          onClick={() => setRiskFilter(riskFilter === 'critical' ? 'all' : 'critical')}
          active={riskFilter === 'critical'}
        />
        <MetricCard
          title="Needs Your Attention"
          value={highCount}
          subtitle="Schedule a check-in this week"
          icon={Building}
          trend="up"
          onClick={() => setRiskFilter(riskFilter === 'high' ? 'all' : 'high')}
          active={riskFilter === 'high'}
        />
        <MetricCard
          title="Monitor Closely"
          value={mediumCount}
          subtitle="Watch for warning signs"
          icon={Activity}
          onClick={() => setRiskFilter(riskFilter === 'medium' ? 'all' : 'medium')}
          active={riskFilter === 'medium'}
        />
        <MetricCard
          title="Healthy Clients"
          value={lowCount}
          subtitle="Keep up the good work"
          icon={Shield}
          onClick={() => setRiskFilter(riskFilter === 'low' ? 'all' : 'low')}
          active={riskFilter === 'low'}
        />
        <MetricCard
          title="Team Members at Risk"
          value={teamPredictions.filter(p => p.risk_level === 'high' || p.risk_level === 'critical').length}
          subtitle="May leave within 90 days"
          icon={Users}
          onClick={() => setRiskFilter('all')}
        />
      </div>

      <Tabs defaultValue="clients" className="space-y-4">
        <TabsList>
          <TabsTrigger value="clients" className="flex items-center gap-2">
            <Building className="h-4 w-4" /> Client Risk
          </TabsTrigger>
          <TabsTrigger value="team" className="flex items-center gap-2">
            <Users className="h-4 w-4" /> Team Risk
          </TabsTrigger>
          <TabsTrigger value="model" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" /> How It Works
          </TabsTrigger>
        </TabsList>

        {/* Filters */}
        <div className="flex gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by name, industry, or role..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={riskFilter} onValueChange={setRiskFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="Risk Level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Risks</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Client Predictions Tab */}
        <TabsContent value="clients" className="space-y-4">
          {/* Risk vs Tenure Scatter Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Client Risk Map</CardTitle>
              <CardDescription>Risk score vs. client tenure — larger dots = more seats at stake</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="tenure_months"
                    name="Tenure"
                    unit=" mo"
                    label={{ value: 'Tenure (months)', position: 'bottom', offset: 5, style: { fontSize: 12 } }}
                  />
                  <YAxis
                    type="number"
                    dataKey="risk_score"
                    name="Risk Score"
                    unit="%"
                    domain={[0, 100]}
                    label={{ value: 'Risk Score', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
                  />
                  <ZAxis type="number" dataKey="current_seat_count" range={[60, 400]} name="Seats" />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ payload }) => {
                      if (!payload || !payload.length) return null;
                      const d = payload[0].payload;
                      return (
                        <div className="bg-popover border rounded-lg p-3 shadow-md text-sm">
                          <p className="font-semibold">{d.client_name}</p>
                          <p className="text-muted-foreground">{d.industry} • {d.region}</p>
                          <div className="mt-1 space-y-0.5">
                            <p>Risk: <span className="font-medium">{d.risk_score}%</span></p>
                            <p>Tenure: <span className="font-medium">{d.tenure_months} months</span></p>
                            <p>Seats: <span className="font-medium">{d.current_seat_count}</span></p>
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Scatter data={filteredClients}>
                    {filteredClients.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={RISK_COLORS[entry.risk_level]} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-3">
              <h3 className="font-semibold text-lg">Client Risk Rankings</h3>
              {filteredClients.map(prediction => (
                <ClientPredictionRow key={prediction.client_id} prediction={prediction} />
              ))}
              {filteredClients.length === 0 && (
                <p className="text-center text-muted-foreground py-8">No clients match filters</p>
              )}
            </div>
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Your Client Health Summary</CardTitle>
                  <CardDescription>At a glance — where to focus today</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-500" />
                        <span className="text-sm font-medium">Act Now</span>
                      </div>
                      <span className="text-sm font-bold text-red-600">{criticalCount} client{criticalCount !== 1 ? 's' : ''}</span>
                    </div>
                    <Progress value={(criticalCount / clientPredictions.length) * 100} className="h-3 [&>div]:bg-red-500" />
                    <p className="text-xs text-muted-foreground">High likelihood of leaving — reach out today</p>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-orange-500" />
                        <span className="text-sm font-medium">Schedule Check-in</span>
                      </div>
                      <span className="text-sm font-bold text-orange-600">{highCount} client{highCount !== 1 ? 's' : ''}</span>
                    </div>
                    <Progress value={(highCount / clientPredictions.length) * 100} className="h-3 [&>div]:bg-orange-500" />
                    <p className="text-xs text-muted-foreground">Showing warning signs — check in this week</p>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-yellow-500" />
                        <span className="text-sm font-medium">Keep Watching</span>
                      </div>
                      <span className="text-sm font-bold text-yellow-600">{mediumCount} client{mediumCount !== 1 ? 's' : ''}</span>
                    </div>
                    <Progress value={(mediumCount / clientPredictions.length) * 100} className="h-3 [&>div]:bg-yellow-500" />
                    <p className="text-xs text-muted-foreground">Minor concerns — monitor over next 30 days</p>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                        <span className="text-sm font-medium">Healthy</span>
                      </div>
                      <span className="text-sm font-bold text-green-600">{lowCount} client{lowCount !== 1 ? 's' : ''}</span>
                    </div>
                    <Progress value={(lowCount / clientPredictions.length) * 100} className="h-3 [&>div]:bg-green-500" />
                    <p className="text-xs text-muted-foreground">Strong engagement — no action needed</p>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Why They Might Leave</CardTitle>
                  <CardDescription>Top reasons for your at-risk clients</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {clientPredictions.filter(p => p.risk_level === 'critical' || p.risk_level === 'high').slice(0, 3).map(p => (
                    <div key={p.client_id} className="space-y-1">
                      <p className="text-sm font-medium">{p.client_name}</p>
                      <ul className="text-xs text-muted-foreground space-y-0.5">
                        {p.top_factors.map((f, i) => (
                          <li key={i}>• {f}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Team Attrition Tab */}
        <TabsContent value="team" className="space-y-4">
          <div className="space-y-3">
            <h3 className="font-semibold text-lg">Team Member Attrition Risk</h3>
            {filteredTeamMembers.map(prediction => (
              <TeamMemberPredictionRow key={prediction.team_member_id} prediction={prediction} />
            ))}
            {filteredTeamMembers.length === 0 && (
              <p className="text-center text-muted-foreground py-8">No team members match filters</p>
            )}
          </div>
        </TabsContent>

        {/* Model Details Tab */}
        <TabsContent value="model" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" /> Model Performance
                </CardTitle>
                <CardDescription>XGBoost Classifier v{modelMetrics.version}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Accuracy</p>
                    <p className="text-xl font-bold">{(modelMetrics.accuracy * 100).toFixed(1)}%</p>
                    <Progress value={modelMetrics.accuracy * 100} className="h-2 mt-1" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">AUC-ROC</p>
                    <p className="text-xl font-bold">{(modelMetrics.auc_roc * 100).toFixed(1)}%</p>
                    <Progress value={modelMetrics.auc_roc * 100} className="h-2 mt-1" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Precision</p>
                    <p className="text-xl font-bold">{(modelMetrics.precision * 100).toFixed(1)}%</p>
                    <Progress value={modelMetrics.precision * 100} className="h-2 mt-1" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Recall (Detection)</p>
                    <p className="text-xl font-bold">{(modelMetrics.recall * 100).toFixed(1)}%</p>
                    <Progress value={modelMetrics.recall * 100} className="h-2 mt-1" />
                  </div>
                </div>
                <div className="pt-4 border-t space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Training Records</span>
                    <span className="font-medium">{modelMetrics.training_records.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Test Records</span>
                    <span className="font-medium">{modelMetrics.test_records}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Trained</span>
                    <span className="font-medium">{modelMetrics.trained_at.split('T')[0]}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Model Type</span>
                    <span className="font-medium">XGBoost Classifier</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" /> Feature Importance
                </CardTitle>
                <CardDescription>What drives churn predictions</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={featureImportance} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 0.4]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <YAxis type="category" dataKey="feature" width={180} tick={{ fontSize: 12 }} />
                    <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                    <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" /> Pipeline Info
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
                <div>
                  <p className="font-medium mb-2">Architecture</p>
                  <ul className="space-y-1 text-muted-foreground">
                    <li>• XGBoost on AWS Lambda</li>
                    <li>• Model stored in S3</li>
                    <li>• EventBridge trigger (daily 2AM UTC)</li>
                    <li>• Predictions → Supabase table</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium mb-2">Data Sources</p>
                  <ul className="space-y-1 text-muted-foreground">
                    <li>• Historical termination forms</li>
                    <li>• Client health scores</li>
                    <li>• Communication frequency</li>
                    <li>• Seat count changes</li>
                  </ul>
                </div>
                <div>
                  <p className="font-medium mb-2">Current Status</p>
                  <ul className="space-y-1 text-muted-foreground">
                    <li>✅ Model trained & validated</li>
                    <li>✅ Local testing complete</li>
                    <li>🔒 Using mock data (safe)</li>
                    <li>⏳ Pending production deployment</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
