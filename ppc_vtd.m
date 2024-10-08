%% Pairwise phase consistency (PPC) function

% Author: Ege Kingir (ege.kingir@med.uni-goettingen.de)

% This function is an implementation of the PPC metric, introduced in Vinck et al. (2010)
% I use PPC for my EEG analysis pipeline for Heartbeat Evoked Potential detection. 
% The goal here is to detect ICA components that significantly involve the volume conduction artifact from the heart.

% PPC is computed between each ICA component or EEG signal and the ECG signal:
%   1) In each trial, phase difference is computed at each time point, giving a phase difference vector.
%   2) Phase difference vectors are summed up, then the length is divided by the number of time points, giving the PPC in that trial.
%   3) Process is repeated for each trial, then the average PPC is computed for the given channel or component.

% At the end of first iteration for PPC calculation, you get all the PPC values as output.
% In accordance with Buot et al. (2021), I detect high PPC components with the mean+3STDs rule.
% Once some components are detected with this manner, I discard that component and detect high PPC components within the remaining ones.
% This iterative approach is ended once 3 components are detected. You can change this in the varargins.

% You can obviously use other time series than EEG and ECG signals. Any time series couple that are recorded simultaneously works!


function [ecg_comps, PPC_all, exclusion_limit] = ppc_vtd(eeg_data, ecg_data, filter, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input: 
%   1) Your EEG data or ICA component data (continuous data, or epoched around an event -- such as the R-peak in my original use case --).
%   2) Your ECG data (epoched in the same way as EEG data).
        %%% NOTE:
        % EEG data: Dimensions are nChannels x time x nTrials
        % ECG data: Dimensions are 1 x time x nTrials
%   3) Do you want your data to be bandpass filtered or not (Binary: 0-no / 1-yes)

%   4) varargin (optional):
%   the order of varargins should be:
%   -lower limit of the bandpass filter (default = 0.1 hz)
%   --upper limit of the bandpass filter (default = 25 hz)
%   ---sampling frequency (default 500 hz)
%   ----maximum number of "high PPC" components that you would like to detect (default = 3)

%%% Output: 
% ecg_comps: the components or channels that have mean+3std PPC value after an iterative evaluation.
% PPC_all: the PPC value between each EEG channel/component and ECG signal.
% exclusion_limit = the PPC value equal to mean+3*SD. If any channel has a PPC value larger than this, it will be excluded.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ecg_comps = [];

% Parameters
nComps = size(eeg_data,1);  % Number of EEG channels or ICA components

if ndims(eeg_data)==3
    nTrials = size(eeg_data,3);
elseif ismatrix(eeg_data) % (i.e. if it is 2 dimensional)
    nTrials=1; % In the case of a single-trial, or continuous data.
end

%% Default values
% Frequency range for filtering
lowFreq = 0.1;  % Low cutoff frequency (0.1 Hz)
highFreq = 25;  % High cutoff frequency (25 Hz)
fs = 500;       % Sampling frequency
num_comp=3;     % number of components or channels that you would like to detect (as having significantly high PPC with the ECG channel)

if isempty(varargin) && filter==1 %user wants filtering with default values
    [b, a] = butter(4, [lowFreq, highFreq] / (fs / 2), 'bandpass');  % 4th-order Butterworth filter
end

%% if varargin is filled
if ~isempty(varargin)
    if length(varargin)==1
        disp('You should not only provide the lower limit for the bandpass, provide the upper limit as well')
        return
    end
    if length(varargin)>3
        num_comp = varargin{4};
    end
    if length(varargin)>2
        fs = varargin{3};
    end
    if length(varargin)==2
        highFreq = varargin{2};
        lowFreq = varargin{1};
    end
end


%% iterative detection of channels or components that have significant similarity to the ECG signal (high PPC)
iterDone=0;
iter=0;

while iterDone==0
    iter=iter+1;
    % Initialize PPC storage for each channel
    PPC_values = [];
    
    % Loop through each channel
    for ch = 1:nComps
        if ismember(ch,ecg_comps)
            continue
        end
        % Store the PPC for each trial for this channel
        PPC_per_trial = zeros(nTrials, 1);
        
        for trial = 1:nTrials
            
            if nTrials>1
                EEG_trial = eeg_data(ch, :, trial); % Get the EEG signal for the current channel and trial 
                ECG_trial = ecg_data(1, :, trial); % Get the ECG signal for the current trial
            elseif nTrials==1
                EEG_trial = eeg_data(ch,:);
                ECG_trial = ecg_data(1,:);
            end
            
            if filter==1
                EEG_filtered = filtfilt(b, a, EEG_trial);
                ECG_filtered = filtfilt(b, a, ECG_trial);
            elseif filter==0
                EEG_filtered = EEG_trial;
                ECG_filtered = ECG_trial;
            end
            
            % Compute the Hilbert transform to extract the analytic signal
            EEG_hilbert = hilbert(EEG_filtered);  % EEG analytic signal
            ECG_hilbert = hilbert(ECG_filtered);  % ECG analytic signal
            
            % Get the phase of the signals
            phase_EEG = angle(EEG_hilbert);    % Phase of the EEG signal
            phase_ECG = angle(ECG_hilbert);    % Phase of the ECG signal
            
            % Convert the phase angles to unit vectors (complex numbers on the unit circle)
            unit_vector_EEG = exp(1i * phase_EEG);  % EEG as unit vector
            unit_vector_ECG = exp(1i * phase_ECG);  % ECG as unit vector
            
            % Compute the difference between the unit vectors
            phase_diff_unit = unit_vector_EEG .* conj(unit_vector_ECG);  % Phase difference on the unit circle
            
            % Compute the pairwise phase consistency (PPC) for this trial
            diff_sum = sum(phase_diff_unit);
            PPC_per_trial(trial) = abs(diff_sum)/length(phase_diff_unit);
        end
        
        % Compute the mean PPC across all trials for this channel
        PPC_values(ch) = mean(PPC_per_trial);
        
    end

    if iter==1
        PPC_all = PPC_values; %before excluding any outlier PPC values, adds all the computed PPC values to be given as output.
    end
      
    for c=1:nComps
        if PPC_values(c) == 0 %this happens starting from the second iteration, in the excluded channels due to >mean+3sd PPC value.
            PPC_values(c) = nan; %let's give them NaN instead of 0 to prevent confusion.
        end
    end
    
    mean_PPC = mean(PPC_values,'omitnan');
    std_PPC = std(PPC_values,'omitnan');
    limit = mean_PPC + 3*std_PPC;

    if iter==1
        exclusion_limit = limit;
    end
    
    ecg_comps = [ecg_comps find(PPC_values>limit)];
    
    if isempty(find(PPC_values>limit)) || length(ecg_comps)>=num_comp
        iterDone=1;
    end
end

end