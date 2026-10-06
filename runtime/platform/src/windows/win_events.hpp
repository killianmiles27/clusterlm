#pragma once
// Translation of Windows power/session notification parameters into platform events. Shared by the service host
// (SERVICE_CONTROL_POWEREVENT / SESSIONCHANGE) and the helper's message window (WM_POWERBROADCAST /
// WM_WTSSESSION_CHANGE), which deliver the same event types and data.
#ifdef _WIN32

#include <windows.h>

#include <optional>

#include "clusterlm/platform/platform_events.hpp"

namespace clusterlm::platform::detail {

// GUID_ACDC_POWER_SOURCE / GUID_POWER_SAVING_STATUS (defined locally: the SDK declares them extern and the
// defining object depends on initguid.h inclusion order).
extern const GUID kGuidAcDcPowerSource;
extern const GUID kGuidPowerSavingStatus;

// `event_type` is PBT_*; `data` is the POWERBROADCAST_SETTING* for PBT_POWERSETTINGCHANGE (else unused).
std::optional<PowerEvent> map_power_broadcast(DWORD event_type, const void* data);
// `event_type` is WTS_*.
std::optional<SessionEvent> map_session_change(DWORD event_type, DWORD session_id);

}  // namespace clusterlm::platform::detail

#endif  // _WIN32
