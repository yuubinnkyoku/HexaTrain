package com.yuubinnkyoku.phonelm

import android.app.Activity
import android.app.Application
import android.os.Bundle
import java.util.concurrent.atomic.AtomicInteger

object HeadlessActivityCounters {
    val create = AtomicInteger()
    val resume = AtomicInteger()
    val becameTop = AtomicInteger()
    val focusTakeover = AtomicInteger()
    fun reset() { create.set(0); resume.set(0); becameTop.set(0); focusTakeover.set(0) }
}

class PhoneLmApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityCreated(activity: Activity, state: Bundle?) { if (activity is MainActivity) HeadlessActivityCounters.create.incrementAndGet() }
            override fun onActivityResumed(activity: Activity) { if (activity is MainActivity) { HeadlessActivityCounters.resume.incrementAndGet(); HeadlessActivityCounters.becameTop.incrementAndGet(); HeadlessActivityCounters.focusTakeover.incrementAndGet() } }
            override fun onActivityStarted(activity: Activity) = Unit
            override fun onActivityPaused(activity: Activity) = Unit
            override fun onActivityStopped(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit
        })
    }
}
